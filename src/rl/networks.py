import torch
from torchtyping import TensorType
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.utils import MODEL_CACHE_DIR

HIDDEN_SIZE = 768 # 768 for bert-base-uncased, distilbert-base-uncased

class encode_network(torch.nn.Module):
    def __init__(self, model_config="bert-base-uncased", freeze_encoder=True, device="cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config, cache_die=MODEL_CACHE_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_config, cache_dir=MODEL_CACHE_DIR).to(device)

        # Freeze transformer encoder
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input: List[str]) -> TensorType['batch_size', HIDDEN_SIZE]:
        input_tokens = self.tokenizer(input, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
        enc = self.model(**input_tokens, output_hidden_states=True)
        # Get last layer hidden states
        last_hidden_states = enc.hidden_states[-1]
        # Get [CLS] hidden states
        sentence_embedding = last_hidden_states[:, 0, :]
        return sentence_embedding

class state_transition_network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 768 for bert-base-uncased, distilbert-base-uncased
        self.gru = torch.nn.GRU(input_size=HIDDEN_SIZE, 
                                hidden_size=HIDDEN_SIZE, 
                                num_layers=1, 
                                batch_first=True,
                                bias=True)
        # Initialize the state transition function to be identity
        self.gru.bias_hh_l0.data[HIDDEN_SIZE:2*HIDDEN_SIZE].fill_(-float('inf'))
        self.gru.bias_ih_l0.data[HIDDEN_SIZE:2*HIDDEN_SIZE].fill_(-float('inf'))

    def forward(self, 
                input: TensorType["batch_size", "seq_len", HIDDEN_SIZE],
                hidden_state: TensorType[1, "batch_size", HIDDEN_SIZE]
        ) -> TensorType["batch", "seq_len", HIDDEN_SIZE]:

        output, _ = self.gru(input, hidden_state)

        return output

class policy_network(torch.nn.Module):
    def __init__(self, action_space=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = torch.nn.Linear(HIDDEN_SIZE, action_space)  

    def forward(self, 
                state: TensorType["batch_size", HIDDEN_SIZE]
        ) -> TensorType["batch_size", "action_space"]:

        x = self.fc1(state)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

class value_network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = torch.nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, 
                state: TensorType["batch_size", HIDDEN_SIZE]
        ) -> TensorType["batch_size", 1]:
        
        x = self.fc1(state)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

class PPOTrainableModelWrapper(torch.nn.Module):
    def __init__(self, policy_model, critic_model, state_transition_model) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.state_transition_model = state_transition_model
    
    def forward(self, input, state):
        new_state = self.state_transition_model(input, state)

        assert new_state.shape[1] == 1, "we use GRU to model the state transition for only one step"
        new_state = new_state.squeeze(1)
        return self.policy_model(new_state), self.critic_model(new_state)
    
    def train(self, mode=True):
        self.policy_model.train(mode)
        self.critic_model.train(mode)
        self.state_transition_model.train(mode)
        
    def eval(self):
        self.policy_model.eval()
        self.critic_model.eval()
        self.state_transition_model.eval()
