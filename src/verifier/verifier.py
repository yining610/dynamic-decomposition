"""Verifier that outputs the factuality results of given claims.
"""
import pickle
import os
import time
import logging

from src.utils.logging_config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

class Verifier(object):
    """Adapated from https://github.com/shmsw25/FActScore/blob/main/factscore/lm.py."""

    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.load_model()

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError

    def format(self, prompt, claim):
        """ format the prompt and claim into a single string for final verification"""
        
        return f"{prompt}\n\nInput: {claim} True or False?\nOutput:"

    def generate(self, prompt, sample_idx=0, max_sequence_length=2048, max_output_length=128):
        prompt = prompt.strip()
        cache_key = f"{prompt}_{sample_idx}"
      
        if self.cache_dict is None:
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)        
        else:
            if cache_key in self.cache_dict:
                logger.info("Reading from verifier cache...")
                return self.cache_dict[cache_key]
            else:
                generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)
                self.cache_dict[cache_key] = generated
                self.add_n += 1

        return generated

    def save_cache(self):
        if self.add_n == 0 or self.cache_file is None:
            return

        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
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
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache

    def get_prob(self):
        """Return the conditional probability of positive label
        """
        raise NotImplementedError