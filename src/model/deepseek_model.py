import os
import logging
import backoff
import openai
import json
from openai import OpenAI

from src.utils.logging_config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

completion_tokens = prompt_cache_miss_tokens = prompt_cache_hit_tokens = 0

class DeepSeekModel:
    def __init__(self, 
                 model: str,
                 temperature: float = 0,
                 max_tokens: int = 1024,
                 n: int = 1,
                 system_setting: str = None
                 ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.system_setting = system_setting

        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

        self.restart()

    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_time=60, max_tries=10)
    def chatcompletions_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def get_response(self, prompt: str):
        global completion_tokens, prompt_cache_miss_tokens, prompt_cache_hit_tokens

        self.message.append({"role": "user", "content": prompt})
        
        try:
            completion = self.chatcompletions_with_backoff(
                model=self.model,
                messages=self.message,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=self.n,
                stream=False
            )
            if completion is None:
                logger.error("Failed to get a valid response from DeepSeek API after retries")
                return ""
        except json.decoder.JSONDecodeError:
            # DeepSeek returns an empty response
            return ""

        completion_tokens += completion.usage.completion_tokens
        prompt_cache_miss_tokens += completion.usage.prompt_cache_miss_tokens
        prompt_cache_hit_tokens += completion.usage.prompt_cache_hit_tokens

        return completion.choices[0].message.content
    
    def __call__(self, prompt):
        return self.get_response(prompt)

    def restart(self):
        self.message = [{"role": "system", "content": self.system_setting}] if self.system_setting else []

    @staticmethod
    def gpt_usage(model="deepseek-chat"):
        global completion_tokens, prompt_cache_miss_tokens, prompt_cache_hit_tokens
        if model == "deepseek-chat": # Currently points to DeepSeek-v3
            cost = completion_tokens / 1000000 * 0.28 + prompt_cache_miss_tokens / 1000000 * 0.14 + prompt_cache_hit_tokens / 1000000 * 0.014
        else:
            raise ValueError(f"Unknown model: {model}")
        
        return {"completion_tokens": completion_tokens, "prompt_cache_miss_tokens": prompt_cache_miss_tokens, "prompt_cache_hit_tokens": prompt_cache_hit_tokens, "cost": cost}
