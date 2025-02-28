from .decomposer import Decomposer
from src.model.proprietary_model import OpenAIModel

class OpenAIDecomposer(OpenAIModel, Decomposer):
    def __init__(
            self, 
            model_handle: str, 
            cache_file: str,
            system_setting: str,
            prompt_format: str,
            temperature: float,
            hostname: str = None,
            port: int = None
        ):

        OpenAIModel.__init__(self, model_handle, system_setting=system_setting, temperature=temperature)
        Decomposer.__init__(self, cache_file, prompt_format=prompt_format, system_prompt=system_setting)

    def _decompose(self, prompt):

        output = self(prompt)[0]
        self.restart()

        return output
    
    def _parse(self, output):

        output = [x.strip() for x in output.split("- ") if x.strip() != ""]

        return output
