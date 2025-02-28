import torch
from tqdm import tqdm
import transformers
from overrides import overrides
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.utils import (
    softmax, 
    convert_model_to_int8_on_gpu, 
    smart_tokenizer_and_embedding_resize, 
    MODEL_CACHE_DIR
)
from .verifier import Verifier

class InstLLamaVerifier(Verifier):
    """Adapted from https://github.com/shmsw25/FActScore/blob/main/factscore/clm.py.
    Follow the README.md to set up the model first.
    """
    def __init__(self, model_handle, model_dir, cache_file, device="cuda"):
        self.model_handle = model_handle
        self.model_dir = model_dir
        self.device = device
        super().__init__(cache_file)

    def recover_instruct_llama(self, output_path, device="cpu", test_recovered_model=False):
        """Heavily adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/weight_diff.py."""

        model_raw = transformers.AutoModelForCausalLM.from_pretrained(
            "huggyllama/llama-7b",
            device_map={"": torch.device(device)},
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            cache_dir=MODEL_CACHE_DIR,
        )
        model_recovered = transformers.AutoModelForCausalLM.from_pretrained(
            "kalpeshk2011/instruct-llama-7b-wdiff",
            device_map={"": torch.device(device)},
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            cache_dir=MODEL_CACHE_DIR,
        )

        tokenizer_raw = transformers.AutoTokenizer.from_pretrained("huggyllama/llama-7b", cache_dir=MODEL_CACHE_DIR)
        if tokenizer_raw.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                model=model_raw,
                tokenizer=tokenizer_raw,
            )
        tokenizer_recovered = transformers.AutoTokenizer.from_pretrained("kalpeshk2011/instruct-llama-7b-wdiff", cache_dir=MODEL_CACHE_DIR)

        state_dict_recovered = model_recovered.state_dict()
        state_dict_raw = model_raw.state_dict()
        for key in tqdm(state_dict_recovered):
            state_dict_recovered[key].add_(state_dict_raw[key])

        if output_path is not None:
            model_recovered.save_pretrained(output_path)
            tokenizer_recovered.save_pretrained(output_path)

        if test_recovered_model:
            input_text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\r\n\r\n"
                "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
            )
            inputs = tokenizer_recovered(input_text, return_tensors="pt")
            out = model_recovered.generate(inputs=inputs.input_ids, max_new_tokens=100)
            output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
            output_text = output_text[len(input_text) :]
            print(f"Input: {input_text}\nCompletion: {output_text}")

        return model_recovered, tokenizer_recovered

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        self.model = convert_model_to_int8_on_gpu(self.model, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

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
                output_scores=True
            )
            gen_tokens = gen_outputs["sequences"]
            
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            gen = gen.split("</s>")[0]
                
            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores
    
    def get_prob(self, scores):
        assert scores.shape[0] in [32000, 32001]
        assert self.tokenizer.decode(5852) == "True" and self.tokenizer.decode(7700) == "False"

        probs = softmax(scores)
        true_prob = probs[5852].item()
        false_prob = probs[7700].item()
        is_supported = true_prob > false_prob

        return is_supported, true_prob, false_prob
    
class LLama3InstVerifier(Verifier):

    def __init__(self, model_handle, model_dir, cache_file, device="cuda"):
        self.model_handle = model_handle
        self.model_dir = model_dir
        self.device = device
        super().__init__(cache_file)

    @overrides
    def format(self, prompt, claim):

        if isinstance(prompt, list):
            prompt_return = prompt
        else:
            prompt_return = [
                    {
                        "role": "system",
                        "content": prompt
                    }
                ]

        prompt_return.append(
            {
                "role": "user",
                "content": "Input: {claim} True or False?\nOutput:".format(
                    claim=claim.strip()
                )
            }
        )

        return self.tokenizer.apply_chat_template(prompt_return, tokenize=False)

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, cache_dir=MODEL_CACHE_DIR).to(self.device)
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
                output_logits=True,
                do_sample=False,
                temperature=0.0,
            )
            gen_tokens = gen_outputs["sequences"]
            # use logits instead of scores as scores all returned inf
            # save the first logits after all system tokens: <|start_header_id|>assistant<|end_header_id|>\n\n
            if len(gen_outputs["logits"]) > 4:
                gen_scores = gen_outputs["logits"][4][0].detach().cpu().numpy()  
            else:
                gen_scores = gen_outputs["logits"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

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
        assert scores.shape[0] in [128256, 128257]
        assert self.tokenizer.decode(2575) == "True" and self.tokenizer.decode(4139) == "False"
        
        probs = softmax(scores)
        true_prob = probs[2575].item()
        false_prob = probs[4139].item()
        is_supported = true_prob > false_prob

        return is_supported, true_prob, false_prob