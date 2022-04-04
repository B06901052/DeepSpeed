"""
This file is models for unit test on unary oprations.
"""
import torch
import torch.nn as nn

# TODO: not finish
"""
test_name: torch_unaryop_test
operation: mean, nanmean, std, 
Testing block for syntax sugar ops which interact with scaler
"""

class Mean(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, inputs):
        torch.mean(inputs, dim=self.dim, keepdim=True)
        return inputs

torch_unaryop_test = {
    "test_name": "torch_unaryop_test",
    "modules": [Mean],
    "input_shape": (200, 300, 400),
    "args": {
            "dim": [0, 1, 2, (0,2), (1,2), -1],
    },
}