import torch
import copy
import random
import logging
import datasets
import numpy as np

from overrides import overrides
from torch.utils.data import IterableDataset
from typing import Tuple, Optional, Union, List, Dict, Any

from src.utils.logging_config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

def whiten(xs: torch.Tensor, shift_mean=True) -> torch.Tensor:
    """Whitens values"""

    var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def pad_sequences(seqs, pad_value, padding='right'):
    """
    Padding sequence to the same length
    """
    assert type(seqs[0]) == list
    max_len = max(len(seq) for seq in seqs)

    if padding == 'right':
        padded_seqs = [seq + [pad_value] * (max_len - len(seq)) for seq in seqs]
    elif padding == 'left':
        padded_seqs = [[pad_value] * (max_len - len(seq)) + seq for seq in seqs]
    else:
        raise ValueError(f"Padding type {padding} not supported.")
    
    return padded_seqs

def print_trainable_params(model: torch.nn.Module):

    parameters = model.parameters()
    num_params = sum(p.numel() for p in parameters if p.requires_grad)

    logger.info(f"Number of trainable parameters: {num_params / 1e6:.2f}M")

class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """

        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()

class IterDataset(IterableDataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return self.size
    
    def sample_generator(self):

        if self.mode == 'train':
            if isinstance(self.data, List):
                random.shuffle(self.data)
            elif isinstance(self.data, datasets.Dataset):
                self.data = self.data.shuffle(42)
            else:
                raise NotImplementedError(f"Data type {type(self.data)} not supported.")

        for sample in self.data:
            yield self.format(sample)

    def batch_generator(self):
        batch = []

        for sample in self.sample_generator():
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        if batch:
            yield batch

    def final_generator(self):
        data_generator = self.batch_generator()
        for batch_sample in data_generator:
            batch = self.batchify(batch_sample)
            yield batch

    def __iter__(self):
        return self.final_generator()
    
class PromptDataset(IterDataset):
    """
    Adapted from: https://github.com/OpenLMLab/MOSS-RLHF/blob/main/ppo/ppo_datahelper.py
    """

    def __init__(self, args, mode='train') -> None:
        super().__init__()
        self.args = args
        self.mode = mode

        self.batch_size = args.rollout_batch_size
        self.data_ratio = args.data_ratio
        data = []
        for data_path in self.args.data_path_list:
            curr_data = datasets.load_from_disk(data_path)[self.mode]
            if self.data_ratio < 1:
                curr_data = curr_data.select(range(int(len(curr_data) * self.data_ratio)))

            data.append(curr_data)
            
        self.data: datasets.Dataset = datasets.concatenate_datasets(data)

        logger.info(f"=============Loaded total {len(self.data)} samples.=============")
        self.size = len(self.data)

    def format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        
        return sample

    def batchify(self, batch_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        return batch_samples

    @overrides
    def batch_generator(self):
        while True:
            for batch in super().batch_generator():
                if len(batch) == self.batch_size:
                    yield batch
            if self.mode != 'train':
                break

class ExperienceDataset(IterDataset):
    """
    Adapted from: https://github.com/OpenLMLab/MOSS-RLHF/blob/main/ppo/ppo_datahelper.py
    """
    def __init__(self, data, args, mode = 'train', **kwargs) -> None:
        self.args = args
        self.mode = mode
        
        self.batch_size = args.mini_batch_size
        self.gamma = args.gamma
        self.lam = args.lam
        self.data: List[Dict[str, Any]] = data
        self.size = len(data)

    def get_advantages_and_returns(self,
        rewards: Union[List[torch.Tensor], torch.Tensor],
        values: List[torch.Tensor],
        use_whitening: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            use_whitening: Whether to use whitening (ie. normalize advantages) or not       
        '''
        if isinstance(rewards, list):
            rewards = torch.tensor(rewards)
        values = torch.tensor(values)

        assert values.size(0) == rewards.size(0), "Values and rewards must have the same length"
        
        episode_length = values.size(0)
        advantages_reversed = []
        lastgaelam = 0
        for t in reversed(range(episode_length)):
            nextvalues = values[t + 1] if t < episode_length - 1 else 0.0
            delta = rewards[t] + self.gamma * nextvalues - values[t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        
        advantages = torch.stack(advantages_reversed[::-1])
        returns = advantages + values

        if use_whitening and advantages.size(0) > 1:
            advantages = whiten(advantages)
        
        return advantages, returns

    def format(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        output = copy.deepcopy(sample)
        advantages, returns = self.get_advantages_and_returns(sample['rewards'], sample['values'], self.args.scale_advantage)

        # original claim length
        output['claim_length'] = len(output['claims_forest'][0].data.split())
        # number of subclaims at the end
        output['num_subclaims'] = len(output['claims_forest'][-1].get_leaves())
        # average subclaim length at the end
        output['avg_subclaim_len'] = np.mean([len(subclaim.split(" ")) for subclaim in output['claims_forest'][-1].get_leaves()])
        # episode length
        output['episode_length'] = len(output['actions'])
        # number of decompositions over one episode
        output['num_decompose'] = np.sum(output['actions'])
        # all claims at each step.
        output['global_claims'] = [tree.get_leaves() for tree in output['claims_forest']]
        
        output['advantages'] = advantages
        output['returns'] = returns
        output['mask'] = torch.ones_like(advantages)

        return output
    
    def batch_generator(self):
        for batch in super().batch_generator():
            yield batch
  
    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:

        batch = {
            key: [sample[key] for sample in batch_samples] for key in batch_samples[0].keys()
        }

        batch['curr_claims'] = pad_sequences([sample['curr_claims'] for sample in batch_samples], pad_value="")
        batch['curr_subclaims'] = pad_sequences([sample['curr_subclaims'] for sample in batch_samples], pad_value=[])
        batch['global_claims'] = pad_sequences([sample['global_claims'] for sample in batch_samples], pad_value=[])

        batch['actions'] = torch.tensor(pad_sequences([sample['actions'] for sample in batch_samples], pad_value=0), dtype=torch.long)   # batch_size x max_episode_length            
        batch['log_probs'] = torch.tensor(pad_sequences([sample['log_probs'] for sample in batch_samples], pad_value=0.))               # batch_size x max_episode_length
        batch['values'] = torch.tensor(pad_sequences([sample['values'] for sample in batch_samples], pad_value=0.))                     # batch_size x max_episode_length

        batch['advantages'] = torch.tensor(pad_sequences([sample['advantages'].tolist() for sample in batch_samples], pad_value=0.))    # batch_size x max_episode_length
        batch['returns'] = torch.tensor(pad_sequences([sample['returns'].tolist() for sample in batch_samples], pad_value=0.))          # batch_size x max_episode_length

        padded_start_states = pad_sequences([sample['start_states'] for sample in batch_samples], pad_value=torch.zeros(768))
        padded_end_states = pad_sequences([sample['end_states'] for sample in batch_samples], pad_value=torch.zeros(768))
        batch['start_states'] = torch.stack([torch.stack(tensors) for tensors in padded_start_states]) # batch_size x max_episode_length x 768
        batch['end_states'] = torch.stack([torch.stack(tensors) for tensors in padded_end_states])     # batch_size x max_episode_length x 768
               
        batch['mask'] = torch.tensor(pad_sequences([sample['mask'].tolist() for sample in batch_samples], pad_value=0.))
        
        return batch