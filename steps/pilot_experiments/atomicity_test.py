import json
import os
import click
import string
import datasets
import logging
from tqdm import tqdm
from typing import Dict, Text, Optional

from src.verifier.verifier import Verifier
from src.verifier.retrieval import Retrieval
from src.utils.prompts import FACTSCORE_VERIFY_PROMPT
from src.utils.utils import save_cache
from src.utils.configs import get_factscore_verification_params
from src.utils.logging_config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

@click.command()
@click.option("--atomicity", type=click.INT, help="atomicity of the given data. Default 0 is the human annotated atomic level", default=0)
@click.option("--policy-name", type=click.STRING, help="Verification policy")
@click.option("--model-name", type=click.STRING, help="Verifier model name.")
@click.option("--write-to", type=click.Path(exists=False, file_okay=False), help="Path to save the verification results.")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), help="Path to save the processed data.")
@click.option("--device", type=click.STRING, help="Device to run the task.", default="cuda")
def main(
    atomicity: int,
    policy_name: str,
    model_name: str,
    write_to: str,
    data_dir: str,
    device: str
):  
    
    if not os.path.exists(write_to):
        os.makedirs(write_to)
    save_path = os.path.join(write_to, f"{model_name}_{policy_name}_{atomicity}.jsonl")
    if os.path.exists(save_path):
        logger.info(f"Verification results already exist at {save_path}.")
        return
    
    logger.info("Preparing data for verification.")

    params = get_factscore_verification_params(
        data_path=os.path.join(data_dir, f"atomicity_{atomicity}"),
        split="test",
        model_name=model_name,
        verify_policy=policy_name,
        device=device
    )

    data: datasets.Dataset = params['data']
    verifier: Optional[Verifier] = params['verifier']
    retrieval: Dict[Text, Retrieval] = params['retrieval']

    knowledge_source = "enwiki-20230401"

    logger.info("Starting verification.")
    results = []
    for atom in tqdm(data):

        if policy_name == "retrieval":
            passages = retrieval[knowledge_source].get_passages(atom['topic'], atom['text'], k=5)
            definition = "Answer the question about {topic} based on the given context.\n\n".format(
                topic=atom['topic']
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
        elif policy_name == "demonstration":
                definition = FACTSCORE_VERIFY_PROMPT 
        elif policy_name == "non-context":
            definition = ""
        else:
            raise ValueError("undefined verification policy.")

        claim_to_verify = verifier.format(prompt=definition, claim=atom['text']).strip()

        generation, score = verifier.generate(claim_to_verify)
        is_supported, true_prob, false_prob = verifier.get_prob(score)

        results.append(
            {
                "is_support": is_supported, 
                "true_prob": true_prob,
                "false_prob": false_prob,
                "generation": generation,
                "text": atom['text'],
                "label": atom['label'] if "label" in atom else None,
                "topic": atom['topic']
            }
        )
    
        if len(results) % 10 == 0:
            save_cache(model_name=f"{model_name}_{policy_name}", verifier=verifier, retrieval=retrieval)

    save_cache(model_name=f"{model_name}_{policy_name}", verifier=verifier, retrieval=retrieval)

    with open(save_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == '__main__':
    main()