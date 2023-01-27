
from torch import nn
from transformers import AutoModel, AutoConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from typing import Dict, Any, List
import torch
import numpy as np


class MultiLayerOutput(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_layers: int, hidden_size: int, activation_function = nn.ReLU(), dropout: float = 0.5):
        nn.Module.__init__(self)
        self.layers = OrderedDict()
        self.layers["input"] = nn.Linear(input_size, hidden_size)
        self.layers["inputdropout"] = nn.Dropout(p=dropout)
        self.layers["inputact"] = activation_function
        for i in num_layers:
            self.layers[f"hidden{i}"] = nn.Linear(hidden_size, hidden_size)
            self.layers[f"hiddendropout{i}"] = nn.Dropout(p=dropout)
            self.layers[f"hiddenact{i}"] = activation_function
        self.layers["output"] = nn.Linear(hidden_size, output_size)
        self.layers["outputdropout"] = nn.Dropout(p=dropout)
        self.layers["output_act"] = activation_function
        self.layers = nn.Sequential(self.layers)
    
    def forward(self, inputs):
        return self.layers(inputs)

class IMPALATransformer(nn.Module, TorchModelV2):

    def __init__(self, 
        obs_space, 
        action_space, 
        num_outputs, 
        model_config, 
        name, 
        model_name_or_path: str = "bert-base-uncased", 
        freeze_encoder: bool = False, 
        output_hidden_size: int = 1024, 
        output_hidden_layers: int = 0, 
        output_dropout: float = 0.5,
        action_embeddings: torch.Tensor = None
        ):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        nn.Module.__init__(self)

        self.config = model_config
        self.encoder_config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.observation_space = obs_space
        self.action_space = action_space

        self.activation = nn.GELU()

        if output_hidden_layers > 0:
            self.action_outputs = MultiLayerOutput(self.encoder_config.hidden_size, num_outputs, output_hidden_layers, output_hidden_size, nn.ReLU(), output_dropout)
            self.value_network = MultiLayerOutput(self.encoder_config.hidden_size, 1, output_hidden_layers, output_hidden_size, nn.ReLU(), output_dropout)
        else:
            self.action_outputs = nn.Linear(self.encoder_config.hidden_size, num_outputs)
            self.value_network = nn.Linear(self.encoder_config.hidden_size, 1)

        self.action_embeddings = None
        if action_embeddings:
            self.action_embeddings = action_embeddings.sum(-2)
        self._features = None

    def forward(self, input_dict: Dict, state: List, seq_lens: Any):

        #print("Input obs size: " + str(input_dict["obs"].size()))
        #print(input_dict["obs"].size())
        encoded = self.encoder(input_ids = input_dict["obs"])

        pooled = torch.mean(encoded[0], 1)
        self._features = pooled

        if self.action_embeddings:
            # 'intent' embedding is [BATCH, PAD_SEQUENCE, HIDDEN]
            #  action embedding is [NUM_ACTIONS, HIDDEN]
            #  result has to be [BATCH, NUM_ACTIONS]
            #  Broadcasting goes through dimensions back to front
            encoded = encoded[0].unsqueeze(0).sum(-2)
            # intent is now [BATCH, 1, HIDDEN]

            # Calc [BATCH, 1, HIDDEN] * [NUM_ACTIONS, HIDDEN]. HIDDEN matches, NUM_ACTIONS and BATCH get broadcasted. 
            # HIDDEN is then reduced with mean to get [BATCH, NUM_ACTIONS]
            logits = (encoded * self.action_embeddings).mean(-1)
        else:
            logits = self.activation(self.action_outputs(pooled))

        return logits, state
        
    def value_function(self):
        assert self._features is not None, "Must call forward first"
        return torch.reshape(self.activation(self.value_network(self._features)), [-1])