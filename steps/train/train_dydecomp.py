import logging
import random
import numpy as np
import torch

from src.utils.arguments import ArgumentParser, TrainArguments, print_args
from src.rl.dydecomp_trainer import PPOTrainer
from src.utils.configs import get_rl_train_params
from src.utils.logging_config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

def main():
    
    parser = ArgumentParser((TrainArguments))
    args = parser.parse() 
    print_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    params = get_rl_train_params(args=args)

    # Initialize the trainer
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

    trainer.train()

    if hasattr(trainer.decomposer, 'gpt_usage'):
        logger.info(trainer.decomposer.gpt_usage())

if __name__ == "__main__": 
    main()
