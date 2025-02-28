"""Prepare FactScore claims on different atomicities for training and evaluation."""
import click
import os
import datasets
import logging
from tqdm import tqdm

from steps.eval.eval_configs import (
    extract_factscore_claims,
    convert_list_to_dict, 
    merge_claims
)

from src.decomposer.openai_decomposer import OpenAIDecomposer 
from src.utils.prompts import FACTSCORE_ATOM_DECOMPOSITION_PROMPT
from src.utils.utils import OUTPUT_CACHE_DIR
from src.utils.logging_config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

@click.command()
@click.option("--data-save-dir", type=str, help="Directory to save the prepared data.")
@click.option("--custom-split-path", type=str, help="Path to the custom split dataset.")
@click.option("--atomicity", type=int, help="Atomicity level for merging or decomposing claims.")
def main(
    data_save_dir: str,
    custom_split_path: str,
    atomicity: int
):
    """modified version of steps/eval/eval_configs.py:prepare_factscore_claims"""
    data_path = os.path.join(data_save_dir, f"atomicity_{atomicity}")
    if os.path.exists(data_path):
        return

    dataset = datasets.load_from_disk(custom_split_path)
    _, train_claims, train_subclaims, _ = extract_factscore_claims(dataset['train'])
    _, val_claims, val_subclaims, _ = extract_factscore_claims(dataset['validation'])
    _, test_claims, test_subclaims, _ = extract_factscore_claims(dataset['test'])

    if atomicity == 0:
        result = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_dict(convert_list_to_dict(train_subclaims)),
                "validation": datasets.Dataset.from_dict(convert_list_to_dict(val_subclaims)),
                "test": datasets.Dataset.from_dict(convert_list_to_dict(test_subclaims))
            }
        )
        result.save_to_disk(data_path)
    elif atomicity > 0:
        result = {}
        for split, claims in zip(["train", "validation", "test"], [train_claims, val_claims, test_claims]):
            # merge claims
            data = {claim['topic']: [] for claim in claims}
            for claim in claims:
                data[claim['topic']].append(claim)
            curr_result = []
            for _, claims in data.items():
                merged_claims, num_atomicity_left = merge_claims(claims, atomicity)
                if num_atomicity_left == 0:
                    curr_result.extend(merged_claims)
                else:
                    continue
                
            if len(curr_result) == 0:
                raise ValueError("atomicity is set too high, no claims to merge.")
            
            result[split] = datasets.Dataset.from_dict(convert_list_to_dict(curr_result))
        
        result = datasets.DatasetDict(result)
        result.save_to_disk(data_path)

    elif atomicity == -1:
        # we don't need atomicity -1 train or validation data, only use test data for pilot experiments
        decomposer = OpenAIDecomposer(model_handle="gpt-3.5-turbo",
                                      cache_file=os.path.join(OUTPUT_CACHE_DIR, "decomposition", "eval", f"gpt-3.5-turbo_factscore-atom.pkl"),
                                      system_setting=FACTSCORE_ATOM_DECOMPOSITION_PROMPT['system'],
                                      prompt_format=FACTSCORE_ATOM_DECOMPOSITION_PROMPT['prompt'],
                                      temperature=0.0)
        result = []
        for data in tqdm(test_subclaims, desc="Decomposing human annotated atomic claims"):
            decomposition = decomposer.decompose(data['text'])
            for subclaim in decomposition:
                result.append({
                    "text": subclaim,
                    "topic": data['topic'],
                    "claim": data['text']
                })
        decomposer.save_cache()
        logger.info(f"Cost: {decomposer.gpt_usage()}")
        result = datasets.DatasetDict({"test": datasets.Dataset.from_dict(convert_list_to_dict(result))})
        result.save_to_disk(data_path)
    else:
        raise ValueError("undefined atomicity value.")

if __name__ == "__main__":
    main()