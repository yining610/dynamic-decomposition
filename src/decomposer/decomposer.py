import os
import time
import pickle
import logging

from src.utils.logging_config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

class Decomposer():
    """Define a HF wrapper class that can decompose a given claim
    """
    def __init__(
            self,
            cache_file: str,
            prompt_format: str = None,
            system_prompt: str = None
        ):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.prompt_format = prompt_format
        self.system_prompt = system_prompt

    def format(self, prompt):
        if self.prompt_format:
            return self.prompt_format.format(claim=prompt)
        return prompt

    def decompose(self, prompt, sample_idx=0):
        prompt = self.format(prompt).strip()
        cache_key = f"{self.system_prompt}_{prompt}_{sample_idx}" if self.system_prompt else f"{prompt}_{sample_idx}"

        if self.cache_dict is None:
            # do not use cache
            generated = self._decompose(prompt)
        else:
            if cache_key in self.cache_dict:
                logger.info("Reading from decomposer cache...")
                generated = self.cache_dict[cache_key]
            else:
                generated = self._decompose(prompt)
                self.cache_dict[cache_key] = generated
                self.add_n += 1

        parsed = self._parse(generated)
        
        return parsed

    def save_cache(self):
        if self.add_n == 0 or self.cache_file is None:
            return

        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=False):
        if self.cache_file is None:
            return None

        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    logger.warning("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache