import backoff
import openai
from openai import OpenAI
import logging

from src.utils.logging_config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()

completion_tokens = prompt_tokens = 0

class OpenAIModel:
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
        self.restart()
        self.client = OpenAI()

    @backoff.on_exception(backoff.expo, openai.OpenAIError)
    def chatcompletions_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    @backoff.on_exception(backoff.expo, openai.OpenAIError)
    def completions_with_backoff(self, **kwargs):
        return self.client.completions.create(**kwargs)

    def chatgpt(self) -> list:
        global completion_tokens, prompt_tokens
        outputs = []
        res = self.chatcompletions_with_backoff(model=self.model, 
                                                messages=self.message, 
                                                temperature=self.temperature, 
                                                max_tokens=self.max_tokens, 
                                                n=self.n)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
        return outputs

    def completiongpt(self) -> list:
        global completion_tokens, prompt_tokens
        outputs = []
        res = self.completions_with_backoff(model=self.model, 
                                            prompt=self.message, 
                                            temperature=self.temperature, 
                                            max_tokens=self.max_tokens, 
                                            n=self.n)
        outputs.extend([choice.text for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
        return outputs

    @staticmethod
    def gpt_usage(model="gpt-4-turbo"):
        global completion_tokens, prompt_tokens
        if model == "gpt-4": # Currently points to gpt-4-0613
            cost = completion_tokens / 1000000 * 60 + prompt_tokens / 1000000 * 30
        elif model == "gpt-4-turbo": # The latest GPT-4 Turbo model with vision capabilities
            cost = completion_tokens / 1000000 * 30 + prompt_tokens / 1000000 * 10
        elif model == "gpt-3.5-turbo": # Currently points to gpt-3.5-turbo-0125
            cost = completion_tokens / 1000000 * 1.5 + prompt_tokens / 1000000 * 0.5
        elif model == "gpt-3.5-turbo-instruct":
            cost = completion_tokens / 1000000 * 2 + prompt_tokens / 1000000 * 1.5
        elif "davinci" in model:
            cost = completion_tokens / 1000000 * 2 + prompt_tokens / 1000000 * 2
        else:
            raise ValueError(f"Unknown model: {model}")
        return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

    def __call__(self, input: str) -> list:
        
        if "davinci" in self.model or "instruct" in self.model:
            self.message = self.message + "\n" + input
            output = self.completiongpt()
        else:
            self.message.append({"role": "user", "content": input})
            output = self.chatgpt()
        return output
        
    def update_message(self, output: str):
        if "davinci" in self.model or "instruct" in self.model:
            self.message = self.message + "\n" + output
        else:
            self.message.append({"role": "assistant", "content": output})
    
    def restart(self):
        if "davinci" in self.model or "instruct" in self.model:
            self.message = self.system_setting if self.system_setting else ""
        else:
            self.message = [{"role": "system", "content": self.system_setting}] if self.system_setting else []