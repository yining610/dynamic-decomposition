"""HF model wrapper that has the
saving property.
"""
import os
import json
import torch
import transformers
from overrides import overrides
from .model import Model
from src.utils.utils import MODEL_CACHE_DIR
DEFAULT_PAD_TOKEN = "<pad>"

class HuggingfaceWrapperModule(Model):
    def __init__(
        self,
        model_handle: str,
    ):
        super().__init__()
        self.model_handle = model_handle
        if "t5" in model_handle.lower():
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.model_handle,
                cache_dir=MODEL_CACHE_DIR
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_handle,
                cache_dir=MODEL_CACHE_DIR
            )

        elif "gpt" in model_handle.lower() or "llama" in model_handle.lower():
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_handle,
                cache_dir=MODEL_CACHE_DIR,
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_handle,
                cache_dir=MODEL_CACHE_DIR
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        else:
            raise NotImplementedError("model is not supported.")
        
    def forward(self, *args, **kwargs):
        """Forward generation of the model.
        """
        model_outputs = self.model(*args, **kwargs)
        # append predictions to the outputs
        return {
            "loss": model_outputs.loss,
            "logits": model_outputs.logits,
            "predictions": model_outputs.logits.argmax(dim=-1)
        }
        
    def generate(self, *args, **kwargs):
        """Generate from the model.
        """
        return self.model.generate(*args, **kwargs)
    
    @overrides
    def save_to_dir(self, path: str):
        """Save the model to the given path.
        """
        self.model.save_pretrained(path)
        
        # save the model_handle as well.
        with open(os.path.join(path, 'custom.config'), 'w', encoding='utf-8') as file_:
            json.dump({
                'model_handle': self.model_handle
            }, file_, indent=4, sort_keys=True)

    @classmethod
    def load_from_dir(cls, path: str) -> "HuggingfaceWrapperModule":
        """Load the model from the given path.
        """
        if os.path.exists(os.path.join(path, 'custom.config')):
            with open(os.path.join(path, 'custom.config'), 'r', encoding='utf-8') as file_:
                config = json.load(file_)
            model_handle = config['model_handle']
        elif os.path.exists(os.path.join(path, 'config.json')):
            with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8') as file_:
                config = json.load(file_)
            model_handle = config['_name_or_path']
        else:
            raise FileNotFoundError("No config file found.") 

        model = cls(model_handle=model_handle)
        if "t5" in model_handle.lower():
            model.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(path)
        elif "gpt" in model_handle.lower() or "llama" in model_handle.lower():
            model.model = transformers.AutoModelForCausalLM.from_pretrained(path)
        else:
            raise NotImplementedError("model is not supported.")
        return model