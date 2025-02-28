"""define a model abstract class of pytorch
models that can be trained.
"""
import torch
import os
from abc import ABC, abstractmethod


class Model(torch.nn.Module, ABC):
    """
    """
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def save_to_dir(self, path: str):
        """
        """
        raise NotImplementedError("Model is an abstract class.")
    
    @classmethod
    def load_from_dir(cls, path: str):
        """
        """
        raise NotImplementedError("Model is an abstract class.")
    
    @classmethod
    def load_from_best(cls, path: str):
        """
        """
        return cls.load_from_dir(os.path.join(path, "best_1"))