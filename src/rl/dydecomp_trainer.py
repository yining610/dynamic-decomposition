import os
import torch
import time
import wandb
import copy
import argparse
import logging
import string
import pickle
import numpy as np

from torch.utils.data import DataLoader
from torchtyping import TensorType
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from accelerate import Accelerator

from src.scorers.scorer import ScorerInstance
from src.scorers.uni_scorer import UNLIConfidenceBoostScorer
from src.decomposer.decomposer import Decomposer
from src.tree.tree import Tree
from src.verifier.verifier import Verifier
from src.metrics.metric import (
    Metrics, 
    MeanMetric, 
    RealtimeMetric, 
    SumMetric
)
from src.optimizer_constructors.optimizer_constructor import OptimizerConstructor
from src.utils.utils import to_device
from src.utils.prompts import FACTSCORE_VERIFY_PROMPT
from src.rl.networks import PPOTrainableModelWrapper
from src.rl.data_configs import (
    RunningMoments,
    ExperienceDataset,
    PromptDataset,
    print_trainable_params
)

from src.utils.logging_config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()


class TrainState:
    def __init__(self):
        self.total_steps = 0
        self.total_examples = 0
        self.best_score = -9999999
    
    def state_dict(self):
        return {
            'total_steps': self.total_steps,
            'total_examples': self.total_exps,
            'best_score': self.best_score,
        }

class PPOTrainer():
    def __init__(
        self,
		args: argparse.Namespace,
        encoder: torch.nn.Module,
        state_transition: torch.nn.Module,
        policy: torch.nn.Module,
        critic: torch.nn.Module,
        initial_state: torch.nn.parameter.Parameter,
        decomposer: Decomposer,
        verifier: Verifier,
        scorer: UNLIConfidenceBoostScorer,
        optimizer_constructor: OptimizerConstructor,
        retrieval: Optional[Dict] = None
    ):
        """Initialize a trainer.
        """
        super().__init__()
        self.args = args
        
		# initialize the models
        self.encoder = encoder
        self.model = PPOTrainableModelWrapper(
            policy_model=policy, 
            critic_model=critic, 
            state_transition_model=state_transition
        )
        self.initial_state = initial_state
        self.decomposer = decomposer
        self.verifier = verifier

        self.scorer = scorer
        self.retrieval = retrieval
        print_trainable_params(model=self.model)

        self.optimizer = optimizer_constructor.construct(self.model)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.args.train_steps,
        )

        self.accelerator = Accelerator()
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        # get unwrapped trainable model
        self.policy = self.accelerator.unwrap_model(self.model).policy_model
        self.critic = self.accelerator.unwrap_model(self.model).critic_model
        self.state_transition = self.accelerator.unwrap_model(self.model).state_transition_model

		# initialize training parameters
        self.data_loader = DataLoader(
            PromptDataset(self.args, mode='train'), 
            batch_size=None, 
            num_workers=self.args.num_workers
        )
        self.train_size = len(self.data_loader.dataset)
        self.data_loader = iter(self.data_loader)

        self.logger = wandb.init(project=args.wandb_project, name=args.wandb_name, config=args) if args.wandb else None

        self.train_metrics = self.build_metrics('train', logger=self.logger)
        self.valid_metrics = self.build_metrics('validation', logger=self.logger)
        self.train_state = TrainState()
        self.running = RunningMoments()
        self.replay_buffer = []

    def build_metrics(self, mode='train', logger=None):
        metrics = Metrics(mode=mode, logger=logger)

        metrics.create_metric('claim_len', MeanMetric())
        metrics.create_metric('episode_len', MeanMetric())
        metrics.create_metric('num_subclaims', MeanMetric())
        metrics.create_metric('avg_subclaim_len', MeanMetric())
        metrics.create_metric('num_decompose', MeanMetric())
        metrics.create_metric('rewards', MeanMetric())            # raw reward for one rollout
        metrics.create_metric('avg_info_loss', MeanMetric())      # average information loss for one rollout
        if mode == 'train':
            metrics.create_metric('lr', RealtimeMetric())
            metrics.create_metric('ups', MeanMetric())
            metrics.create_metric('loss', MeanMetric())
            metrics.create_metric('reward_mean', MeanMetric())    # real-time reward mean
            metrics.create_metric('reward_std', MeanMetric())     # real-time reward std
            metrics.create_metric('global_num_examples', SumMetric())
            metrics.create_metric('values', MeanMetric())
            metrics.create_metric('values_error', MeanMetric())
            metrics.create_metric('returns', MeanMetric())
            metrics.create_metric('advantages', MeanMetric())
            metrics.create_metric('ratio', MeanMetric())
            metrics.create_metric('padding_percentage', MeanMetric())
            metrics.create_metric('pg_clipfrac', MeanMetric())
            metrics.create_metric('vf_clipfrac', MeanMetric())
            metrics.create_metric('pg_loss', MeanMetric())
            metrics.create_metric('vf_loss', MeanMetric())
            metrics.create_metric('entropy_loss', MeanMetric())


        return metrics
    
    def policy_model_forward(self, state):
        return self.policy(state)

    def value_model_forward(self, state):
        return self.critic(state)

    def state_transition_forward(self, input, hidden_state):
        return self.state_transition(input, hidden_state)
    
    def PPO_model_forward(self, input, state):
        return self.model(input, state)

    def get_avg_info_loss(self, subclaims: List[str], claim: str, topic: str) -> float:
        """Compute the average inforamtion loss after decomposing the claim"""

        if len(subclaims) == 0:
            return 0.0 # no decomposition at all, no information loss

        subclaim_instances = [ScorerInstance(text=subclaim, topic=topic) for subclaim in subclaims]
        claim_instance = ScorerInstance(text=claim, topic=topic)

        # the higher the score, the less fine-grained the decomposition
        subclaim_scores: List[float] = self.scorer(subclaim_instances)
        claim_score: float = self.scorer(claim_instance)

        # the smaller the difference, subclaims are more closer to the atomic level.
        avg_info_loss = sum([claim_score - subclaim_score for subclaim_score in subclaim_scores]) / len(subclaim_scores)

        return avg_info_loss

    def get_verification_confidence(self, claim: str, topic: str=None) -> Tuple[float]:
        """Compute the verification confidence of the given claim."""

        knowledge_source = "enwiki-20230401"

        if self.args.verify_policy == "retrieval":
            passages = self.retrieval[knowledge_source].get_passages(topic, claim, k=5)
            definition = "Answer the question about {topic} based on the given context.\n\n".format(
                topic=topic
            )
    
            context = ""
            for psg_idx, psg in enumerate(reversed(passages)):
                context += "Title: {title}\nText: {text}\n\n".format(
                    title=psg["title"], 
                    text=psg["text"].replace("<s>", "").replace("</s>", "")
                )
            definition += context.strip()
            if not definition[-1] in string.punctuation:
                definition += "."
        elif self.args.verify_policy == "demonstration":
            definition = FACTSCORE_VERIFY_PROMPT
        elif self.args.verify_policy == "non-context":
            definition = ""
        else:
            raise ValueError("undefined verification policy.")

        claim_to_verify = self.verifier.format(prompt=definition, claim=claim).strip()

        _, score = self.verifier.generate(claim_to_verify)
        is_supported, true_prob, false_prob = self.verifier.get_prob(score)
        
        return abs(true_prob - false_prob), is_supported

    def get_reward(self, action: int, claim: str, subclaims: List[str], topic: str) -> torch.Tensor:
        """Compute the reward, which is the verification confidence change after decomposition."""

        if action == 0 or len(subclaims) == 0:
            # Policy chooses not to decompose 
            # or the given decomposer failed to decompose the claim even if the policy chooses to decompose
            return torch.tensor(0.) # no decomposition, no confidence change
        else:
            before_decompose_confidence = self.get_verification_confidence(claim, topic)[0]
            after_decompose_confidence_avg = sum([self.get_verification_confidence(subclaim, topic)[0] for subclaim in subclaims]) / len(subclaims) 

            return torch.tensor(after_decompose_confidence_avg - before_decompose_confidence)
    
    def bfs_decomposition(self, 
                          topic: str,
                          claims_tree: Tree,
                          claims_to_decompose: deque, 
                          t: int, 
                          ep_t: int,
                          trajectory: Dict[str, Any]
    ):
        """Recursively perform BFS decomposition on the given claim and return the trajectory."""
        # base case
        if len(claims_to_decompose) == 0 or ep_t > self.args.max_timesteps_per_episode:
            return t, ep_t, trajectory

        t += 1
        ep_t += 1

        # get action
        if len(trajectory['end_states']) == 0:
            start_state = self.initial_state
        else:
            start_state = trajectory['end_states'][-1]

        action_dist = torch.softmax(self.policy_model_forward(start_state.to(self.accelerator.device)), dim=-1)
        action = torch.distributions.Categorical(action_dist).sample()
        log_prob = torch.log(action_dist[action])

        assert action in [0, 1], f"Invalid action. Expected 0 or 1, got {action}"
        if action == 1:
            # perform decomposition
            curr_claim = claims_to_decompose.popleft()
            subclaims: List[str] = self.decomposer.decompose(curr_claim)
            claims_to_decompose.extend(subclaims)
            # update claims in the decomposition tree
            curr_claim_tree = claims_tree.bfs_search(curr_claim)
            for subclaim in subclaims:
                 curr_claim_tree.add_child(Tree(subclaim))
        else:
            # no decomposition
            curr_claim = claims_to_decompose.popleft()
            subclaims = []

        # update state
        all_claim_list = claims_tree.get_leaves()
        all_claim_str = self.encoder.tokenizer.sep_token.join(all_claim_list)
        claim_embedding = self.encoder(all_claim_str) 
        avg_info_loss = torch.tensor(self.get_avg_info_loss(subclaims, curr_claim, topic), device=self.accelerator.device)
        hidden_state = claim_embedding * (1 + torch.sigmoid(avg_info_loss))

        hidden_state = hidden_state.unsqueeze(0)               # 1 x 1 (batch_size) x hidden_size
        start_state = start_state.unsqueeze(0).unsqueeze(0)    # 1 (batch_size) x 1 x hidden_size
        input = {
            "input": start_state,
            "hidden_state": hidden_state
        }
        to_device(input, self.accelerator.device)
        end_state = self.state_transition_forward(**input)
        end_state = end_state.squeeze()                        # hidden_size
        start_state = start_state.squeeze()                    # hidden_size

        # get reward
        reward = self.get_reward(action, curr_claim, subclaims, topic) # torch.Size([])

        # get value
        value = self.value_model_forward(end_state).squeeze()          # torch.Size([])

        # record the trajectory
        curr_claim_tree = copy.deepcopy(claims_tree)
        trajectory['claims_forest'].append(curr_claim_tree)
        trajectory['curr_claims'].append(curr_claim)
        trajectory['curr_subclaims'].append(subclaims)
        trajectory['actions'].append(action.cpu())
        trajectory['start_states'].append(start_state.cpu())
        trajectory['end_states'].append(end_state.cpu())
        trajectory['rewards'].append(reward)
        trajectory['log_probs'].append(log_prob.cpu())
        trajectory['values'].append(value.cpu())
        trajectory['avg_info_loss'].append(avg_info_loss.cpu())

        # recursively decompose the subclaims
        return self.bfs_decomposition(topic, claims_tree, claims_to_decompose, t, ep_t, trajectory)
    
    @torch.no_grad()
    def rollout(self):
        """Run a rollout to collect samples and fill replay buffer."""

        self.model.eval()

        start_time = time.time()
        t = 0
        while t < self.args.timesteps_per_batch:
            batch: List[Dict[str, Any]] = next(self.data_loader)
            for item in batch:
                topic = item['topic']
                claim = item['text']

                if len(claim.split()) > self.args.split_threshold and self.args.heuristic_split:
                    claims = claim.split(". ")
                    claims_tree = Tree(claim)
                    for c in claims:
                        claims_tree.add_child(Tree(c))
                    claims_to_decompose = deque(claims)
                else:
                    claims_tree = Tree(claim)
                    claims_to_decompose = deque([claim])

                ep_t = 0
                trajectory = {
                    "claims_forest": [],    # decomposition tree of each timestep
                    "curr_claims": [],      # current claim before decomposition
                    "curr_subclaims": [],   # current subclaims after decomposition
                    "actions": [],
                    "rewards": [], 
                    "start_states": [], 
                    "end_states": [],
                    "values": [],
                    "log_probs": [],
                    "avg_info_loss": []
                }

                t_pre = t

                t, ep_t, trajectory = self.bfs_decomposition(
                    topic, claims_tree, claims_to_decompose, t, ep_t, trajectory
                )

                rewards = torch.tensor(trajectory['rewards'])
                self.train_metrics.record_metric_many('rewards', rewards.tolist())
                self.train_metrics.record_metric_many('avg_info_loss', torch.tensor(trajectory['avg_info_loss']).tolist())

                if self.args.filter_reward:
                    if torch.mean(rewards) < self.args.filterreward_threshold:
                        logger.info(f"Drop the trajectory with reward: {torch.mean(rewards)}")
                        t = t_pre
                        continue

                if self.args.scale_reward:
                    # Reward scaling
                    rewards_mean, rewards_std = self.running.update(rewards)
                    if self.args.use_reward_norm:
                        rewards = (rewards - self.running.mean) / self.running.std
                    else:
                        rewards /= self.running.std # do not -= mean since advantage will be normalized again
                    logger.info(f"Running reward mean: {self.running.mean}, std: {self.running.std}")
                    self.train_metrics.record_metric('reward_mean', rewards_mean)
                    self.train_metrics.record_metric('reward_std', rewards_std)
                
                if self.args.clip_reward:
                    # Reward clip
                    rewards = torch.clip(rewards, -self.args.cliprange_reward, self.args.cliprange_reward)

                trajectory['rewards'] = rewards
                trajectory['topic'] = topic

                self.replay_buffer.append(trajectory)
            
            if hasattr(self.decomposer, 'gpt_usage'):
                logger.info(f"=========== Current Rollout Progress: {t}/{self.args.timesteps_per_batch}. Current Cost: {self.decomposer.gpt_usage()} ===========")
            else:
                logger.info(f"=========== Current Rollout Progress: {t}/{self.args.timesteps_per_batch} ===========")

        logger.info(f'Rollout {len(self.replay_buffer)} samples with total {t} timesteps in {(time.time() - start_time):.2f} seconds.')
        self.model.train()

    def loss(
        self,
        logprobs: TensorType["batch_size", "episode_size"],
        values: TensorType["batch_size", "episode_size"],
        entropy: TensorType["batch_size", "episode_size"],
        old_logprobs: TensorType["batch_size", "episode_size"],
        old_values: TensorType["batch_size", "episode_size"],
        advantages: TensorType["batch_size", "episode_size"],
        returns: TensorType["batch_size", "episode_size"],
        mask: TensorType["batch_size", "episode_size"]
    ):
        """PPO objective function.
        Adapted from :
        - https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """

        n = mask.sum()

        if self.args.clip_critic_loss:
            # clip value function loss
            values_clipped = torch.clamp(
                values,
                old_values - self.args.cliprange_value,
                old_values + self.args.cliprange_value,
            )
            
            vf_loss1 = (values - returns) ** 2
            vf_loss2 = (values_clipped - returns) ** 2
            vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
            vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n
        else:
            vf_loss = torch.sum((values - returns) * mask) / n
            vf_clipfrac = torch.tensor(0.0)

        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)

        pg_loss1 = advantages * ratio
        pg_loss2 = advantages * torch.clamp(
            ratio,
            1.0 - self.args.cliprange_policy,
            1.0 + self.args.cliprange_policy,
        )

        pg_loss = -torch.sum(torch.min(pg_loss1, pg_loss2) * mask) / n    # to maximize the objective function
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

        entropy_loss = -torch.sum(entropy * mask) / n                     # to maximize the entropy

        loss = pg_loss + self.args.vf_coef * vf_loss + self.args.ent_coef * entropy_loss

        self.train_metrics.record_metric('loss', loss.item())
        self.train_metrics.record_metric('pg_loss', pg_loss.item())
        self.train_metrics.record_metric('vf_loss', vf_loss.item())
        self.train_metrics.record_metric('entropy_loss', entropy_loss.item())
        self.train_metrics.record_metric('pg_clipfrac', pg_clipfrac.item())
        self.train_metrics.record_metric('vf_clipfrac', vf_clipfrac.item())

        self.train_metrics.record_metric('values', torch.sum(values * mask) / n)
        self.train_metrics.record_metric('values_error', torch.sum(((values - returns) * mask) ** 2) / n)
        self.train_metrics.record_metric('advantages', torch.sum(advantages * mask) / n)
        self.train_metrics.record_metric('returns', torch.sum(returns * mask) / n)
        self.train_metrics.record_metric('ratio', torch.sum(ratio * mask) / n)
        self.train_metrics.record_metric('padding_percentage', 1 - n / mask.numel())

        return loss

    def train_step(self, batch: Dict[str, Any]):
        self.optimizer.zero_grad()
        # forward
        assert self.model.training

        mask: TensorType['batch_size', "max_episode_length"] = batch['mask'].to(self.accelerator.device)

        start_states: TensorType['batch_size', 'max_episode_length', 'hidden_size'] = batch['start_states']
        global_claims_batch: List[List[str]] = [[self.encoder.tokenizer.sep_token.join(t) for t in episode] for episode in batch['global_claims']]

        global_decomposition_emb: TensorType['batch_size', 'max_episode_length', 'hidden_size'] = torch.stack(
            [
                self.encoder(episode) for episode in global_claims_batch
            ], 
            dim=0
        )

        # For padded timesteps, the local_info_loss is 0
        local_info_loss: TensorType['batch_size', "max_episode_length", 1]= torch.stack(
            [
                torch.tensor(
                    [
                        self.get_avg_info_loss(
                            subclaims=sub_claims_episode[i],
                            claim=claim_episode[i],
                            topic = topic
                        )  for i in range(len(sub_claims_episode))
                    ],
                    dtype=torch.float
                ) for sub_claims_episode, claim_episode, topic in zip(batch['curr_subclaims'], batch['curr_claims'], batch['topic'])
            ],
            dim=0
        ).unsqueeze(-1).to(self.accelerator.device)

        hidden_states = global_decomposition_emb * (1 + torch.sigmoid(local_info_loss))

        policy_output, critic_output = [], []
        for i in range(start_states.size(1)):
            curr_start_state = start_states[:, i, :].contiguous().unsqueeze(1)
            curr_hidden_state = hidden_states[:, i, :].contiguous().unsqueeze(1)
            # swap batch_size and D∗num_layers
            curr_hidden_state = torch.swapaxes(curr_hidden_state, 0, 1)
            inputs = {
                "input": curr_start_state,
                "state": curr_hidden_state
            }
            to_device(inputs, self.accelerator.device)
            p_out, c_out = self.PPO_model_forward(**inputs)
            policy_output.append(p_out)
            critic_output.append(c_out)

        policy_output = torch.stack(policy_output, dim=1)         # batch_size x max_episode_length x action_space
        critic_output = torch.stack(critic_output, dim=1)         
        critic_output = critic_output.squeeze(-1) * mask          # batch_size x max_episode_length

        log_probs_dist = torch.log_softmax(policy_output, dim=-1) # batch_size x max_episode_length x action_space
        entropy = -torch.sum(torch.exp(log_probs_dist) * log_probs_dist, dim=-1) * mask
        log_probs = torch.gather(log_probs_dist, 
                                 dim=-1, 
                                 index=batch['actions'].unsqueeze(-1).to(self.accelerator.device)
                                ).squeeze(-1) * mask
        
        # compute loss
        loss_input = {
            "logprobs": log_probs,
            "values": critic_output,
            "entropy": entropy,
            "old_logprobs": batch['log_probs'],
            "old_values": batch['values'],
            "advantages": batch['advantages'],
            "returns": batch['returns'],
            "mask": mask
        }
        to_device(loss_input, self.accelerator.device)
        loss = self.loss(**loss_input)

        self.accelerator.backward(loss)

        if torch.isnan(loss) or torch.isinf(loss) or loss.abs().gt(10000.):
            logger.warning(f'Strange loss {loss.item()} detected.')

        self.optimizer.step()
        if not self.accelerator.optimizer_step_was_skipped:
            self.scheduler.step(loss)

    @torch.no_grad() 
    def evaluate(self, mode='validation'):
        logger.info("Start evaluation")
        self.model.eval()
        start_time = time.time()

        valid_dataloader = DataLoader(
            PromptDataset(self.args, mode=mode), 
            batch_size=None, 
            num_workers=self.args.num_workers
        )

        decomposition_result = {"claims": [], "subclaims": [], "labels": [], "topics": []}
        
        for step, batch in enumerate(valid_dataloader):
            for item in batch:
                topic = item['topic']
                claim = item['text']

                if len(claim.split()) > self.args.split_threshold and self.args.heuristic_split:
                    claims = claim.split(". ")
                    claims_tree = Tree(claim)
                    for c in claims:
                        claims_tree.add_child(Tree(c))
                    claims_to_decompose = deque(claims)
                else:
                    claims_tree = Tree(claim)
                    claims_to_decompose = deque([claim])
            
                ep_t, t = 0, 0
                trajectory = {
                    "claims_forest": [],  
                    "curr_claims": [],    
                    "curr_subclaims": [], 
                    "actions": [],
                    "rewards": [], 
                    "start_states": [], 
                    "end_states": [],
                    "values": [],
                    "log_probs": [],
                    "avg_info_loss": []
                }

                t, ep_t, trajectory = self.bfs_decomposition(
                    topic, claims_tree, claims_to_decompose, t, ep_t, trajectory
                )

                self.valid_metrics.record_metric_many('rewards', torch.tensor(trajectory['rewards']).tolist())
                self.valid_metrics.record_metric_many('avg_info_loss', torch.tensor(trajectory['avg_info_loss']).tolist())
                self.valid_metrics.record_metric('episode_len', ep_t)
                self.valid_metrics.record_metric('claim_len', len(trajectory['claims_forest'][0].data.split()))
                self.valid_metrics.record_metric('num_subclaims', len(trajectory['claims_forest'][-1].get_leaves()))
                self.valid_metrics.record_metric('avg_subclaim_len', np.mean([len(subclaim.split(" ")) for subclaim in trajectory['claims_forest'][-1].get_leaves()]))
                self.valid_metrics.record_metric('num_decompose', np.sum(trajectory['actions']))    
                
                # for testing
                decomposition_result['topics'].append(item['topic'])
                decomposition_result['claims'].append(item['text'])
                decomposition_result['subclaims'].append(trajectory['claims_forest'][-1].get_leaves())
                if 'label' in item:
                    decomposition_result['labels'].append(item['label'])

        metrics = self.valid_metrics.all_gather_metrics()
        self.valid_metrics.display(self.train_state.total_steps, gathered_metrics=metrics, eval=True)
        self.valid_metrics.write_wandb(self.train_state.total_steps, gathered_metrics=metrics)

        validation_score = metrics['rewards']
        self.valid_metrics.reset(no_reset=[])
        
        logger.info(f'Evaluation completed in {(time.time() - start_time):.2f} seconds.')
        self.model.train()
        torch.cuda.empty_cache()

        return validation_score, decomposition_result

    def train(self):

        logger.info('Start training')
        self.model.train()
        while self.train_state.total_steps < self.args.train_steps:
            # sample new replay buffer
            self.rollout()
            self.decomposer.save_cache()
            self.verifier.save_cache()
            # sample mini-batches 
            self.train_loader = DataLoader(
                ExperienceDataset(self.replay_buffer, self.args), 
                batch_size=None, 
                num_workers=self.args.num_workers
            )

            for batch in self.train_loader:
                if self.train_state.total_steps >= self.args.train_steps:
                    break
                
                start_time = time.time()

                with torch.no_grad():
                    batchsize = len(batch['rewards'])
                    self.train_state.total_examples += batchsize

                    self.train_metrics.record_metric('global_num_examples', batchsize)
                    self.train_metrics.record_metric_many('episode_len', batch['episode_length'])
                    self.train_metrics.record_metric_many('claim_len', batch['claim_length'])
                    self.train_metrics.record_metric_many('num_subclaims', batch['num_subclaims'])
                    self.train_metrics.record_metric_many('avg_subclaim_len', batch['avg_subclaim_len'])
                    self.train_metrics.record_metric_many('num_decompose', batch['num_decompose'])

                # perform a step of train
                self.train_step(batch)
                del batch
                
                # record
                cost_time = time.time() - start_time
                self.train_metrics.record_metric('ups', 1. / cost_time)
                if hasattr(self.scheduler, 'get_last_lr'):
                    lr = self.scheduler.get_last_lr()[0]
                else:
                    lr = self.optimizer.param_groups[0]['lr']
                self.train_metrics.record_metric('lr', lr)
                self.train_state.total_steps += 1
                
                need_reset = False
                if self.train_state.total_steps % self.args.logging_interval == 0:
                    metrics = self.train_metrics.all_gather_metrics()
                    self.train_metrics.display(self.train_state.total_steps, self.train_size, gathered_metrics=metrics)
                    self.train_metrics.write_wandb(self.train_state.total_steps, gathered_metrics=metrics)
                    need_reset = True
                    
                # do evaluation for every save_per_step steps
                if self.train_state.total_steps % self.args.save_per_step == 0:
                    eval_score, _ = self.evaluate()
                    self.model.train()
                        
                    # save checkpoint
                    is_best = eval_score > self.train_state.best_score
                    if is_best:
                        self.train_state.best_score = eval_score
                        logger.info(f'Greater than the best score {abs(eval_score)}.')
                    else:
                        logger.info(f'Did not beat the best score {abs(self.train_state.best_score)}.')
                        
                    self.save_checkpoint(is_best=is_best, total_steps=self.train_state.total_steps)
        
                if need_reset:
                    self.train_metrics.reset()

            self.train_loader = None
            self.replay_buffer.clear()
            torch.cuda.empty_cache()

    def save_checkpoint(self, is_best: bool, total_steps: int):
        os.makedirs(self.args.model_save_path, exist_ok=True)
        best_model_path = os.path.join(self.args.model_save_path, 'best_model')
        steps_model_path = os.path.join(self.args.model_save_path, 'Steps_{}'.format(total_steps))

        # get unwrapped trainable model
        unwrapped_policy_model = self.accelerator.unwrap_model(self.model).policy_model
        unwrapped_critic_model = self.accelerator.unwrap_model(self.model).critic_model
        unwrapped_state_transition_model = self.accelerator.unwrap_model(self.model).state_transition_model

        if is_best:
            torch.save(
                {
                    'policy_state_dict': unwrapped_policy_model.state_dict(),
                    'critic_state_dict': unwrapped_critic_model.state_dict(),
                    'state_transition_state_dict': unwrapped_state_transition_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, best_model_path
            )
            logger.info(f'Saved best model to {best_model_path}.')

        torch.save(
            {
                'policy_state_dict': unwrapped_policy_model.state_dict(),
                'critic_state_dict': unwrapped_critic_model.state_dict(),
                'state_transition_state_dict': unwrapped_state_transition_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, steps_model_path
        )
        logger.info(f'Saved model of {total_steps} steps to {steps_model_path}.')
        
def test_data_loader(trainer: PPOTrainer):
    batch: List[Dict[str, Any]] = next(trainer.data_loader)
    logger.info(batch[0].keys())
    assert len(batch) == trainer.args.rollout_batch_size

def test_bfs_decomposition(trainer: PPOTrainer):
    batch = next(trainer.data_loader)
    item = batch[0]
    topic = item['topic']
    claim = item['text']
    claims_tree = Tree(claim)
    claims_to_decompose = deque([claim])
    ep_t, t = 0, 0
    trajectory = {
        "claims_forest": [],    # decomposition tree of each timestep
        "curr_claims": [],      # current claim before decomposition
        "curr_subclaims": [],   # current subclaims after decomposition
        "actions": [],
        "rewards": [], 
        "start_states": [], 
        "end_states": [],
        "values": [],
        "log_probs": [],
        "avg_info_loss": []
    }

    t, ep_t, trajectory = trainer.bfs_decomposition(
        topic, claims_tree, claims_to_decompose, t, ep_t, trajectory
    )

    logger.info(claim)
    logger.info(f"timesteps: {t}, episode length: {ep_t}")
    print("current subclaims: ", trajectory['curr_subclaims'])
    print("actions: ", trajectory['actions'])
    print("rewards: ", trajectory['rewards'])
    print("values: ", trajectory['values'])
    print("log probs: ", trajectory['log_probs'])
    trajectory['claims_forest'][-1].print_tree()

    logger.info(trainer.decomposer.gpt_usage())

def test_loss(trainer: PPOTrainer):
    if os.path.exists("replay_buffer.pkl"):
        with open("./replay_buffer.pkl", "rb") as f:
            trainer.replay_buffer = pickle.load(f)
    else:
        trainer.rollout()
        with open("./replay_buffer.pkl", "wb") as f:
            pickle.dump(trainer.replay_buffer, f)

    trainer.train_loader = DataLoader(
        ExperienceDataset(trainer.replay_buffer, trainer.args), 
        batch_size=None, 
        num_workers=trainer.args.num_workers
    )
    for batch in trainer.train_loader:

        batchsize = len(batch['rewards'])
        trainer.train_state.total_examples += batchsize

        trainer.train_metrics.record_metric('global_num_examples', batchsize)
        trainer.train_metrics.record_metric_many('episode_len', batch['episode_length'])
        trainer.train_metrics.record_metric_many('claim_len', batch['claim_length'])
        trainer.train_metrics.record_metric_many('num_subclaims', batch['num_subclaims'])
        trainer.train_metrics.record_metric_many('avg_subclaim_len', batch['avg_subclaim_len'])
        trainer.train_metrics.record_metric_many('num_decompose', batch['num_decompose'])

        trainer.train_step(batch)
        del batch
    
    trainer.train_metrics.display(trainer.train_state.total_steps)

if __name__ == "__main__":
    from src.utils.configs import get_rl_train_params 
    from src.utils.arguments import ArgumentParser, TrainArguments, print_args

    parser = ArgumentParser((TrainArguments))
    args = parser.parse() 
    print_args(args)

    params = get_rl_train_params(args=args)

    trainer = PPOTrainer(
        args=args,
        encoder=params['encoder'],
        state_transition=params['state_transition'],
        policy=params['policy'],
        critic=params['critic'],
        initial_state=params['init_state'],
        decomposer=params['decomposer'],
        verifier=params['verifier'],
        scorer=params['scorer'],
        optimizer_constructor=params['optimizer_constructor'],
        retrieval=params['retrieval']
    )

    test_data_loader(trainer)
    test_bfs_decomposition(trainer)
    test_loss(trainer)