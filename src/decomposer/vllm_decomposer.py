from typing import Union
from .decomposer import Decomposer
from src.model.vllm_model import VLLMModel, VLLMServer

NONSENSE_RESPONSES = ["Here is the decomposed claim:", 
                      "Here is the decomposition of the claim into two sub-claims:",
                      "Please provide the claim you'd like me to decompose. I'll follow the principles to break it down into two sub-claims.",
                      "Here are the decomposed claims:",
                      "Here is the decomposition of the claim:"]

class VLLMDecomposerModel(VLLMModel, Decomposer):
    def __init__(
        self,
        model_handle: str,
        cache_file: str,
        system_setting: str,
        prompt_format: str,
        temperature: float,
        hostname: str = None,
        port: int = None,
    ):
        # hostname, port are ignored here—but kept for a unified signature
        VLLMModel.__init__(self, model_handle, system_setting, temperature)
        Decomposer.__init__(self, cache_file, prompt_format, system_setting)

    def _decompose(self, prompt):
        output = self(prompt)
        self.restart()
        return output

    def _parse(self, output):
        output = [x.strip() for x in output.split("- ") if x.strip() != ""]
        for remove_str in NONSENSE_RESPONSES:
            output = [x.replace(remove_str, "").strip() for x in output]
        output = [x for x in output if x != ""]
        return output

class VLLMDecomposerServer(VLLMServer, Decomposer):
    def __init__(
        self,
        model_handle: str,
        cache_file: str,
        system_setting: str,
        prompt_format: str,
        temperature: float,
        hostname: str = None,
        port: int = None,
    ):
        VLLMServer.__init__(self, model_handle, system_setting, temperature, hostname=hostname, port=port)
        Decomposer.__init__(self, cache_file, prompt_format, system_setting)

    def _decompose(self, prompt):
        output = self(prompt)
        self.restart()
        return output

    def _parse(self, output):
        output = [x.strip() for x in output.split("- ") if x.strip() != ""]
        
        for remove_str in NONSENSE_RESPONSES:
            output = [x.replace(remove_str, "").strip() for x in output]
        output = [x for x in output if x != ""]
        return output

class VLLMDecomposer:
    def __new__(
        cls,
        model_handle: str,
        cache_file: str,
        system_setting: str,
        prompt_format: str,
        temperature: float,
        hostname: str = None,
        port: int = None,
    ):
        if hostname:
            real_cls = VLLMDecomposerServer
        else:
            real_cls = VLLMDecomposerModel

        instance = super(VLLMDecomposer, cls).__new__(real_cls)
        # Call the appropriate __init__ on that new object
        instance.__init__(model_handle, cache_file, system_setting, prompt_format, temperature, hostname, port)
        return instance