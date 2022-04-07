"""
This file is models for unit test on unary oprations.
"""
from turtle import forward
import torch
import torch.nn as nn

# TODO: not finish
"""
test_name: torch_unaryop_test
operation:
    pointwise: abs
    reduction: mean, std, 
Testing block for syntax sugar ops which interact with scaler
"""

class Abs(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.abs(inputs)
    
class Absolute(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.absolute(inputs)
    
class Acos(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.acos(inputs)
    
class Arccos(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.arccos(inputs)
    
class Acosh(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.acosh(inputs)
    
class Arccosh(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.arccosh(inputs)
    
class Asin(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.asin(inputs)
    
class Arcsin(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.arcsin(inputs)
    
class Asinh(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.asinh(inputs)
    
class Arcsinh(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.arcsinh(inputs)
    
class Atan(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.atan(inputs)
    
class Arctan(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.arctan(inputs)
    
class Atanh(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.atanh(inputs)
    
class Arctanh(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.arctanh(inputs)
    
class Ceil(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.ceil(inputs)

class Clamp(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.clamp(inputs, 0, 100)
    
class Clip(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return torch.clip(inputs, 0, 100)

class Unary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
class Argmax(Unary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        torch.argmax(inputs, dim=self.dim)
        return inputs
    
class Argmin(Unary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        torch.argmin(inputs, dim=self.dim)
        return inputs

class Amax(Unary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        torch.amax(inputs, dim=self.dim)
        return inputs
    
class Amin(Unary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        torch.amin(inputs, dim=self.dim)
        return inputs

class Aminmax(Unary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        torch.aminmax(inputs, dim=self.dim)
        return inputs

class max(Unary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        torch.max(inputs, dim=self.dim)
        return inputs
    
class min(Unary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        torch.min(inputs, dim=self.dim)
        return inputs

class Sum(Unary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        torch.sum(inputs, dim=self.dim)
        return inputs
    

class Mean(Unary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        torch.mean(inputs, dim=self.dim)
        return inputs

class Correction(nn.Module):
    def __init__(self, dim=-1, unbiased=None, correction=None):
        super().__init__()
        self.dim = dim
        self.unbiased = unbiased
        self.correction = correction
        
    @classmethod
    def check_args_valid(cls, *args, **kwargs):
        return (kwargs.get("unbiased") is None or
                kwargs.get("correction") is None)

class Var(Correction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        kwargs = dict()
        if self.unbiased is not None:
            kwargs["unbiased"] = self.unbiased
        if self.correction is not None:
            kwargs["correction"] = self.correction
        torch.var(inputs, dim=self.dim, **kwargs)
        return inputs

class VarMean(Correction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        kwargs = dict()
        if self.unbiased is not None:
            kwargs["unbiased"] = self.unbiased
        if self.correction is not None:
            kwargs["correction"] = self.correction
        torch.var_mean(inputs, dim=self.dim, **kwargs)
        return inputs

class Std(Correction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        kwargs = dict()
        if self.unbiased is not None:
            kwargs["unbiased"] = self.unbiased
        if self.correction is not None:
            kwargs["correction"] = self.correction
        torch.std(inputs, dim=self.dim, **kwargs)
        return inputs

class StdMean(Correction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, inputs):
        kwargs = dict()
        if self.unbiased is not None:
            kwargs["unbiased"] = self.unbiased
        if self.correction is not None:
            kwargs["correction"] = self.correction
        torch.std_mean(inputs, dim=self.dim, **kwargs)
        return inputs
    
torch_math_pointwiseop_test = {
    "test_name": "torch_math_pointwiseop_test",
    "modules": [Abs, Absolute, Acos, Arccos, Acosh, Arccosh, Asin, Arcsin, Asinh, Arcsinh, Atan, Arctan, Atanh, Arctanh, Ceil, Clamp, Clip],
    "input_shape": (3, 4, 5),
    "args": {},
}


torch_math_reduction_correctionop_test = {
    "test_name": "torch_reduction_correctionop_test",
    "modules": [Var, VarMean, Std, StdMean],
    "input_shape": (3, 4, 5),
    "args": {
            "dim": [0, 1, 2, (0,2), (1,2), -1],
            "unbiased": [True, False, None],
            "correction": [0, 1, 2, None],
    },
}
