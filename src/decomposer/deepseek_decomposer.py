from .decomposer import Decomposer
from src.model.deepseek_model import DeepSeekModel

NONSENSE_RESPONSES = ["No additional subclaim is needed",
                      "No additional subclaim can be generated",
                      "No additional subclaim needed",
                      "No second subclaim is needed",
                      "No further decomposition is possible",
                      "No further decomposition possible",
                      "No further decomposition needed",
                      "no further decomposition is needed",
                      "The claim is already concise",
                      "The original claim is too short",
                      "The original claim is already concise",
                      "This claim cannot be further decomposed"]

class DeepSeekDecomposer(DeepSeekModel, Decomposer):
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

        DeepSeekModel.__init__(self, model_handle, system_setting=system_setting, temperature=temperature)
        Decomposer.__init__(self, cache_file, prompt_format=prompt_format, system_prompt=system_setting)

    def _decompose(self, prompt):

        output = self(prompt)
        self.restart()

        return output
    
    def _parse(self, output):

        for remove_str in NONSENSE_RESPONSES:
            if remove_str.lower() in output.lower():
                return []

        output = [x.strip() for x in output.split("- ") if x.strip() != ""]
        return output
