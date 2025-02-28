import os
import logging
import torch
import datasets
from typing import Dict, Any

from src.optimizer_constructors.optimizer_constructor import AdamWConstructor

from src.verifier.llama_verifier import (
    InstLLamaVerifier,
    LLama3InstVerifier
)
from src.entailers.entailer import Entailer
from src.entailers.soft_entailer import SoftEntailer
from src.scorers.uni_scorer import UNLIConfidenceBoostScorer
from src.verifier.t5_verifier import T5Verifier
from src.verifier.retrieval import DocDB, Retrieval
from src.decomposer.vllm_decomposer import VLLMDecomposer
from src.decomposer.deepseek_decomposer import DeepSeekDecomposer
from src.rl.networks import (
    encode_network, 
    state_transition_network, 
    policy_network,
    value_network
)

from .logging_config.logging_config import setup_logging
from .prompts import * 

from .utils import (
    DATABASE_DIR, 
    OUTPUT_CACHE_DIR
)

__DECOMPOSITION_POLICIES__ = {
    "factscore": FACTSCORE_DECOMPOSITION_PROMPT,
    "factscore-atom": FACTSCORE_ATOM_DECOMPOSITION_PROMPT,
    "binary": BINARY_SPLIT_DECOMPOSITION_PROMPT,
    "wice": WICE_PROMPT,
    "rnd": RND_PROMPT
}

__DECOMPOSER_NAME_TO_CLASSES__ = {
    "llama3-70b-inst-4bit": {'cls': VLLMDecomposer, 'model_dir': "casperhansen/llama-3-70b-instruct-awq"},
    "deepseek-chat": {'cls': DeepSeekDecomposer, 'model_dir': "deepseek-chat"}
}

__VERIFIER_NAME_TO_CLASSES__ = {
    "inst-llama-7B": {'cls': InstLLamaVerifier, 'model_dir': "YOUR_LOCAL_PATH_TO_LLAMA_7B"},
    "llama3-8b-inst": {'cls': LLama3InstVerifier, 'model_dir': "meta-llama/Meta-Llama-3-8B-Instruct"},
    "ft-t5-3B": {'cls': T5Verifier, 'model_dir': "ylu610/FT-T5-3B-Verifier"}
}

setup_logging()
logger = logging.getLogger()

def register_knowledge_source(
        name="enwiki-20230401", 
        policy="retrieval",
        data_dir=DATABASE_DIR,
        cache_dir=OUTPUT_CACHE_DIR,
        batch_size=256, # batch size for retrieval,
        device="cuda"
    ):
    retrieval, db = {}, {}
    db_path = os.path.join(data_dir, f"{name}.db")
    data_path = os.path.join(data_dir, f"{name}.jsonl")
    cache_path = os.path.join(cache_dir, "verification", f"retrieval-{name}.json")
    embed_cache_path = os.path.join(cache_dir, "verification", f"retrieval-{name}.pkl")
    db[name] = DocDB(db_path=db_path, data_path=data_path)
    retrieval[name] = Retrieval(
            db=db[name], 
            cache_path=cache_path, 
            embed_cache_path=embed_cache_path, 
            batch_size=batch_size,
            device=device
    )

    return retrieval

def get_rl_train_params(
    args: Dict[str, Any]
):

    logger.info(f"Loading decomposer: {args.decomposer_name}")
    decomposer_config = __DECOMPOSER_NAME_TO_CLASSES__[args.decomposer_name]
    decomposer = decomposer_config['cls'](
        model_handle=decomposer_config['model_dir'],
        cache_file=os.path.join(OUTPUT_CACHE_DIR, "decomposition", args.mode, f"{args.decomposer_name}_{args.decompose_policy}.pkl") if args.cache_decomposition else None,
        system_setting=__DECOMPOSITION_POLICIES__[args.decompose_policy]['system'],
        prompt_format=__DECOMPOSITION_POLICIES__[args.decompose_policy]['prompt'],
        temperature=args.temperature,
        hostname=args.hostname,
        port=args.port
    )

    logger.info(f"Loading verifier: {args.verifier_name}")
    verifier_config = __VERIFIER_NAME_TO_CLASSES__[args.verifier_name]
    verifier = verifier_config['cls'](
        model_handle=args.verifier_name,
        model_dir=verifier_config['model_dir'],
        cache_file=os.path.join(OUTPUT_CACHE_DIR, "verification", args.mode, f"{args.verifier_name}_{args.verify_policy}.pkl") if args.cache_verification else None,
        device=args.device
    )

    retrieval = register_knowledge_source(name="enwiki-20230401", policy=args.verify_policy, device=args.device)

    logger.info(f"Loading PPO networks")
    encoder = encode_network(model_config="bert-base-uncased", freeze_encoder=True, device=args.device)
    init_state= torch.nn.parameter.Parameter(torch.zeros(768), requires_grad=False)
    state_transition = state_transition_network()
    policy = policy_network()
    critic = value_network()

    logger.info(f"Loading scorer")

    # bleached claims are designed for biography verification task 
    scorer = UNLIConfidenceBoostScorer(
        bleached_templates=[
            "{topic} is a person.",
            "{topic} breathes.",
            "{topic} exists.",
            "{topic} is a name.",
            "{topic} is unique.",
            "{topic} is famous.",
            "{topic} has some abilities.",
            "somebody knows {topic}.",
            "{topic} is a star."
        ],
        entailer=SoftEntailer(
            model_name="Zhengping/roberta-large-unli",
            device=args.device,
            internal_batch_size=256,
            max_length=256
        ),
        cap_entailer=Entailer(
            model_name="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            device=args.device,
            internal_batch_size=256,
            max_length=256
        )
    )

    optimizer_constructor = AdamWConstructor(learning_rate=args.lr)
    
    return {
        "decomposer": decomposer,
        "verifier": verifier,
        "retrieval": retrieval,
        "encoder": encoder,
        "init_state": init_state,
        "state_transition": state_transition,
        "policy": policy,
        "critic": critic,
        'scorer': scorer,
        "optimizer_constructor": optimizer_constructor
    }

def get_factscore_verification_params(
    data_path: str,
    split: str,
    model_name: str,
    verify_policy: str,
    device: str="cuda"
):
    
    dataset = datasets.load_from_disk(data_path)[split]

    verifier_config = __VERIFIER_NAME_TO_CLASSES__[model_name]
    verifier = verifier_config['cls'](
        model_handle=model_name,
        model_dir=verifier_config['model_dir'],
        cache_file=os.path.join(OUTPUT_CACHE_DIR, "verification", "baselines", f"{model_name}_{verify_policy}.pkl"),
        device=device
    )
    
    retrieval = register_knowledge_source(name="enwiki-20230401", policy=verify_policy, device=device)

    return {
        "verifier": verifier,
        "retrieval": retrieval,
        "data": dataset
    }