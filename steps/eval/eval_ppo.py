import logging
import os
import json
import torch

from src.rl.dydecomp_trainer import PPOTrainer 
from src.rl.networks import policy_network, state_transition_network, value_network
from src.utils.configs import get_rl_train_params
from src.utils.arguments import ArgumentParser, TrainArguments, EvalArguments, print_args
from src.utils.logging_config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

def main():
    parser = ArgumentParser((TrainArguments, EvalArguments))
    train_args, eval_args = parser.parse() 
    print_args(eval_args)

    # load model
    logger.info("=========== Loading Models ===========")
    checkpoint = torch.load(eval_args.saved_model_path, weights_only=True)
    policy = policy_network()
    state_transition = state_transition_network()
    critic = value_network()
    policy.load_state_dict(checkpoint['policy_state_dict'])
    state_transition.load_state_dict(checkpoint['state_transition_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])

    # load params
    params = get_rl_train_params(args=train_args)
    params['policy'] = policy
    params['state_transition'] = state_transition
    params['critic'] = critic # will not be used during decomposition

    train_args.data_path_list = [eval_args.test_data_path]

    trainer = PPOTrainer(
        args=train_args,
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

    result_save_dir = os.path.join(eval_args.result_save_dir, f"de_{train_args.decomposer_name}_ve_{train_args.verifier_name}_{train_args.decompose_policy}_{train_args.verify_policy}_{eval_args.tag}")
    os.makedirs(result_save_dir, exist_ok=True)
    file_name = f"test_atomicity_{eval_args.atomicity}"
    decompose_save_path = os.path.join(result_save_dir, f"{file_name}_decomposition.json")

    logger.info("=========== Start Decomposition ===========")
    if os.path.exists(decompose_save_path):
        decomposition_result = json.load(open(decompose_save_path, "r"))
    else:
        _, decomposition_result = trainer.evaluate(mode="test")
        with open(decompose_save_path, "w") as f:
            json.dump(decomposition_result, f, indent=4)

    logger.info("=========== Start Verification ===========")
    verify_save_path = os.path.join(result_save_dir, f"{file_name}_verification.json")
    if os.path.exists(verify_save_path):
        pass
    else:
        verification_result = {'claims': {"prediction": [], "confidence": []}, "subclaims": {"prediction": [], "confidence": []}, "labels": []}
        for claim, subclaims, label, topic in zip(decomposition_result['claims'], decomposition_result['subclaims'], decomposition_result['labels'], decomposition_result['topics']):
            claim_conf, claim_pred = trainer.get_verification_confidence(claim, topic)
            verification_result['claims']['prediction'].append(claim_pred)
            verification_result['claims']['confidence'].append(claim_conf)

            subclaim_conf_list, subclaim_pred_list = [], []
            for subclaim in subclaims:
                subclaim_conf, subclaim_pred = trainer.get_verification_confidence(subclaim, topic)
                subclaim_pred_list.append(subclaim_pred)
                subclaim_conf_list.append(subclaim_conf)
            verification_result['subclaims']['prediction'].append(subclaim_pred_list)
            verification_result['subclaims']['confidence'].append(subclaim_conf_list)

            verification_result['labels'].append(label)
        with open(verify_save_path, "w") as f:
            json.dump(verification_result, f, indent=4)

    trainer.decomposer.save_cache()
    trainer.verifier.save_cache()
    if hasattr(trainer.decomposer, "gpt_usage"):
        logger.info(trainer.decomposer.gpt_usage())
    logger.info("=========== Finish Evaluation ===========")
    
if __name__ == "__main__":
    main()