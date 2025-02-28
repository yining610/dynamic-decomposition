import os
import json
import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.utils import softmax, MODEL_CACHE_DIR
from .verifier import Verifier

class T5Verifier(Verifier):
    def __init__(self, model_handle, model_dir, cache_file=None, device="cuda"):
        self.model_handle = model_handle
        self.model_dir = model_dir
        self.device = device
        super().__init__(cache_file)

    def load_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir, cache_dir=MODEL_CACHE_DIR).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, cache_dir=MODEL_CACHE_DIR)

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]

        input_ids = self.tokenizer(prompts).input_ids
        if verbose:
            input_ids = tqdm(input_ids)

        generations = []
        scores = []
        for curr_input_ids in input_ids:
            if len(curr_input_ids) > max_sequence_length - max_output_length:
                curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
            curr_input_ids = torch.LongTensor([curr_input_ids]).to(self.device)
            gen_outputs = self.model.generate(
                curr_input_ids,
                max_length=curr_input_ids.shape[1]+max_output_length,
                return_dict_in_generate=True,
                output_scores=True,
                num_return_sequences=1
            )
            gen_tokens = gen_outputs["sequences"]
            # assert len(gen_outputs["scores"]) == gen_outputs["sequences"].shape[-1] - 1
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, :])

            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)
                
            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores
    
    def get_prob(self, scores):
        assert scores.shape[0] in [32128, 32129]
        assert self.tokenizer.decode(10998) == "True" and self.tokenizer.decode(10747) == "Fal"
        
        probs = softmax(scores)
        true_prob = probs[10998].item() 
        false_prob = probs[10747].item()
        is_supported = true_prob > false_prob

        return is_supported, true_prob, false_prob