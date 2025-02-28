import dataclasses
import os
import sys

from dataclasses import dataclass, field
from typing import Any, List, NewType, Optional, Tuple

from transformers import HfArgumentParser
DataClassType = NewType("DataClassType", Any)

@dataclass
class TrainArguments:
    """
    Customized arguments for PPO training
    """
    mode: str = field(
        default="train",
        metadata={
            "help": "Task mode.",
            "choices": ["train", "eval"]
        },
    )

    # model arguments
    decomposer_name: str = field(
        default="llama3-70b-inst-4bit",
        metadata={
            "help": (
                "Decomposer model name or path"
                "Check out src/utils/configs.py __DECOMPOSER_NAME_TO_CLASSES__ for available models."
            )
        },
    )
    verifier_name: str = field(
        default="llama3-8b-inst",
        metadata={
            "help": (
                "Verifier model name or path"
                "Check out src/utils/configs.py __VERIFIER_NAME_TO_CLASSES__ for available models."
            )
        },
    )
    decompose_policy: str = field(
        default="binary",
        metadata={
            "help": (
                "Decomposition policy used by decomposer."
                "Check out src/utils/configs.py __DECOMPOSITION_POLICIES__ for available policies."
            ),
        },
    )
    verify_policy: str = field(
        default="retrieval",
        metadata={
            "help": "Verification policy used by verifier.",
            "choices": ["retrieval", "demonstration", "non-context"]
        },
    )
    
    # data arguments
    data_path_list: List[str] = field(
        default=None,
        metadata={"help": "List of data paths containing training, validation, and test data."},
    )
    num_workers: int = field(
        default=1,
        metadata={"help": "Number of workers for data loading."},
    )
    num_prefetch: int = field(
        default=32,
        metadata={"help": "Number of prefetch for data loading."},
    )

    # logging arguments
    wandb: bool = field(
        default=True,
        metadata={"help": "Whether to use wandb for logging."},
    )
    wandb_project: str = field(
        default=None,
        metadata={"help": "Wandb project name."},
    )
    wandb_name: str = field(
        default=None,
        metadata={"help": "Wandb run name."},
    )
    logging_interval: int = field(
        default=100,
        metadata={"help": "Number of training steps to log."},
    )

    # training arguments
    device: str = field(
        default="cuda",
        metadata={"help": "Device used for training."},
    )
    train_steps: int = field(
        default=100,
        metadata={"help": "Number of training steps. In other words, number of times to run the rollout and update replay buffer."},
    )
    timesteps_per_batch: int = field(
        default=512,
        metadata={"help": "Max number of timesteps to run per rollout. Use with rollout_batch_size."},
    )
    max_timesteps_per_episode: int = field(
        default=40,
        metadata={"help": "Max number of timesteps per episode."},
    )
    rollout_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for sampling from environment."},
    )
    mini_batch_size: int = field(
        default=4,
        metadata={"help": "Mini batch size for training. *NOT* for sampling from environment."},
    )
    lr: float = field(
        default=3e-5,
        metadata={"help": "Learning rate for policyu and critic model."},
    )
    gamma: float = field(
        default=0.99,
        metadata={"help": "PPO discount factor used for GAE."},
    )
    lam: float = field(
        default=0.95,
        metadata={"help": "PPO lambda used for GAE."},
    )
    clip_critic_loss: bool = field(
        default=True,
        metadata={"help": "Whether to clip critic loss / value."},
    )
    scale_advantage: bool = field(
        default=False,
        metadata={"help": "Whether to scale advantage."},
    )
    filter_reward: bool = field(
        default=False,
        metadata={"help": "Whether to filter out trajectory based on reward value."},
    )
    filterreward_threshold: float = field(
        default=0.0,
        metadata={"help": "Threshold for filtering out trajectory if filter_reward=True."},
    )
    scale_reward: bool = field(
        default=False,
        metadata={"help": "Whether to scale reward."},
    )
    use_reward_norm: bool = field(
        default=False,
        metadata={"help": "Whether to normalize reward using running mean and std when scaling reward (use_reward_scaling=True)."},
    )
    clip_reward: bool = field(
        default=False,
        metadata={"help": "Whether to clip reward."},
    )
    cliprange_value: float = field(
        default=None,
        metadata={"help": "Range for clipping critic loss."},
    )
    cliprange_policy: float = field(
        default=0.2,
        metadata={"help": "Range for clipping policy loss."},
    )
    cliprange_reward: float = field(
        default=None,
        metadata={"help": "Range for clipping reward."}
    )
    vf_coef: float = field(
        default=1.0,
        metadata={"help": "value function loss coefficient."}
    )
    ent_coef: float = field(
        default=0.01,
        metadata={"help": "entropy loss coefficient."}
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature decomposer used for sampling"},
    )
    heuristic_split: bool = field(
        default=False,
        metadata={"help": "Whether to split claim by sentences first before decomposing."},
    )
    split_threshold: int = field(
        default=40,
        metadata={
            "help": (
                "Threshold of claim length to split claim by sentences first if heuristic_split=True.",
                "default is 40 which is the average number of tokens of atomicity level 2"
            )
        },
    )
    model_save_path: str = field(
        default=None,
        metadata={"help": "Path to save model checkpoints."},
    )
    save_per_step: int = field(
        default=100,
        metadata={"help": "Save ckpt per steps"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    hostname: str = field(
        default=None,
        metadata={"help": "Hostname for VLLM server."},
    )
    port: int = field(
        default=None,
        metadata={"help": "Port for VLLM server."},
    )
    cache_decomposition: bool = field(
        default=True,
        metadata={"help": "Whether to cache and read decompositions during training."},
    )
    cache_verification: bool = field(
        default=True,
        metadata={"help": "Whether to cache and read verifications during training."},
    )
    data_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of data used for training."},
    )

@dataclass
class EvalArguments:
    test_data_path: str = field(
        default=None,
        metadata={"help": "Path to test data."},
    )
    saved_model_path: str = field(
        default=None,
        metadata={"help": "Path to saved model."},
    )
    result_save_dir: str = field(
        default="results/FactScore/ChatGPT",
        metadata={"help": "dir to save evaluation results."},
    )
    tag: str = field(
        default="main",
        metadata={"help": "current experiment tag name"}
    )
    atomicity: int = field(
        default=0,
        metadata={"help": "Atomicity level of a claim. The higher the atomicity, the more complex the claim. 0 represents the human annotated atomicity level."},
    )

class ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type is bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | Tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output

def print_args(args):
    """Print arguments."""

    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)