import openai
import backoff
from openai import OpenAI
from vllm import LLM, SamplingParams

from src.utils.utils import MODEL_CACHE_DIR

class VLLMModel:
    def __init__(self,
                 model_handle: str,
                 system_setting: str=None,
                 temperature: str=0.0):
        
        self.model_handle = model_handle
        self.system_setting = system_setting
        self.temperature = temperature
        self.load_model()
        self.restart()

    def load_model(self):
        # We quantize the model for efficiency
        self.model = LLM(model=self.model_handle, 
                         download_dir=MODEL_CACHE_DIR, 
                         tensor_parallel_size=1,
                         quantization="AWQ"
                        )
        
        self.tokenizer = self.model.get_tokenizer()

        self.config = SamplingParams(
            n=1,
            temperature=self.temperature,
            max_tokens=1024,
            stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        )
    
    def __call__(self, prompt):
        if "inst" in self.model_handle.lower():
            self.message.append({"role": "user", "content": prompt})
            self.message = self.tokenizer.apply_chat_template(self.message, tokenize=False)
        else:
            self.message = self.message + "\n" + prompt
        result = self.model.generate([self.message], sampling_params=self.config, use_tqdm=False)
        return result[0].outputs[0].text

    def restart(self):
        if "inst" in self.model_handle.lower():
            self.message = [{"role": "system", "content": self.system_setting}] if self.system_setting else []
        else:
            self.message = self.system_setting if self.system_setting else ""

class VLLMServer:
    def __init__(self, 
                 model_handle: str, 
                 system_setting: str=None,
                 temperature: str=0.0,
                 hostname: str="ta-a6k-001.crc.nd.edu",
                 port: int=8888):
        self.model_handle = model_handle
        self.system_setting = system_setting
        self.temperature = temperature

        self.client = OpenAI(
            base_url=f"http://{hostname}:{str(port)}/v1"
        )
        self.restart()
    
    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_time=60)
    def chatcompletions_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def get_response(self, prompt: str):
        self.message.append({"role": "user", "content": prompt})

        completion = self.chatcompletions_with_backoff(
            model=self.model_handle,
            messages=self.message,
            temperature=self.temperature,
            max_tokens=1024,
            n=1
        )

        return completion.choices[0].message.content
    
    def __call__(self, prompt):
        
        return self.get_response(prompt)

    def restart(self):
        self.message = [{"role": "system", "content": self.system_setting}] if self.system_setting else []
