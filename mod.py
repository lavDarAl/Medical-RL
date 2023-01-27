from torch import nn
from transformers import AutoModel, AutoConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from typing import Dict, Any, List
import torch
import numpy as np



class DQN(object):
    def __init__(self,state_size,
                      action_size,
                      session,
                      summary_write = None,
                      )

class DQNetwork(nn.Module):
    def __init__(self, obs_space: GymDict, act_space: Discrete, num_outputs: int, 
                        model_config: Dict, name: str):
        super().__init__(obs_space, act_space, num_outputs, model_config, name)
        self.internal_model = FullyConnectedNetwork(obs_space, act_space, num_outputs,
            model_config, name + '_internal',
        )
        self.final_layer = tf.keras.layers.Dense(act_space.n, name='q_values', activation=None)
    def forward(self, input_dict, state, seq_lens):
        logits, _ = self.internal_model({'obs': input_dict['obs_flat']})
        q_values = self.final_layer(logits)
        self._value = tf.math.reduce_max(masked_q_values, axis=1)
        return q_values, state

    def value_function(self):
        return self._value