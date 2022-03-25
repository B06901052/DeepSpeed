import torch
import torch.nn as nn

class ScalerOpBlock(nn.Module):
    def __init__(self, dtype, pre):
        super().__init__()
        if dtype == int:
            self.a = 10
        elif dtype == float:
            self.a = 10.0
        elif dtype == torch.Tensor:
            self.a = torch.Tensor([10])
        
        self.pre = pre
            
class Add(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        return self.a + inputs if self.pre else inputs + self.a
    
class Sub(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        return self.a - inputs if self.pre else inputs - self.a
    
class Mul(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        return self.a * inputs if self.pre else inputs * self.a
    
class Div(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        return self.a / inputs if self.pre else inputs / self.a

class UnitsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.int_test = nn.Sequential(
            Add(dtype=int),
            Sub(dtype=int),
            Mul(dtype=int),
            Div(dtype=int),
            Add(dtype=int, pre=False),
            Sub(dtype=int, pre=False),
            Mul(dtype=int, pre=False),
            Div(dtype=int, pre=False),
        )
        self.float_test = nn.Sequential(
            Add(dtype=int),
            Sub(dtype=int),
            Mul(dtype=int),
            Div(dtype=int),
            Add(dtype=int, pre=False),
            Sub(dtype=int, pre=False),
            Mul(dtype=int, pre=False),
            Div(dtype=int, pre=False),
        )
        self.tensor_test = nn.Sequential(
            Add(dtype=int),
            Sub(dtype=int),
            Mul(dtype=int),
            Div(dtype=int),
            Add(dtype=int, pre=False),
            Sub(dtype=int, pre=False),
            Mul(dtype=int, pre=False),
            Div(dtype=int, pre=False),
        )
        
    def forward(self, inputs):
        out = self.int_test(inputs)
        out = self.float_test(inputs)
        out = self.tensor_test(inputs)
        
        return out
