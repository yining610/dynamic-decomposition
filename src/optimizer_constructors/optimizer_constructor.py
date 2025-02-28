from transformers import AdamW, Adafactor

class OptimizerConstructor():
    def __init__(
        self,
        learning_rate: float
    ):
        """
        """
        super().__init__()
        self.learning_rate = learning_rate
        
    def construct(self, model):
        """
        """
        raise NotImplementedError

class AdamWConstructor(OptimizerConstructor):
    def __init__(
        self,
        learning_rate: float,
    ):
        """
        """
        super().__init__(learning_rate=learning_rate)
        
    def construct(self, model):
        """
        """
        return AdamW(
            params=model.parameters(),
            lr=self.learning_rate
        )
    

class AdaFactorConstructor(OptimizerConstructor):
    def __init__(
        self,
        learning_rate: float,
    ):
        """
        """
        super().__init__(learning_rate=learning_rate)
        
    def construct(self, model):
        """
        """
        return Adafactor(
            params=model.parameters(),
            lr=self.learning_rate,
            scale_parameter=False, 
            relative_step=False
        )