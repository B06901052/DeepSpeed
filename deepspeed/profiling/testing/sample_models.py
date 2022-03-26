"""
This file is models for unit test.
"""

import torch
import torch.nn as nn

# %% [markdown]
# ## Something special
# * syntax_sugar_op_test
#   * __rdiv__ will call __mul__ in it, the op count will twice. (not a bug)
#   * __rpow__ and __pow__ both call torch.pow, so just count op in torch.pow

class ScalerOpBlock(nn.Module):
    inplace=False
    def __init__(self, dtype, pre):
        super().__init__()
        if dtype == int:
            self.a = 10
        elif dtype == float:
            self.a = 10.0
        elif dtype == torch.Tensor:
            self.a = torch.Tensor([10])
        
        self.pre = pre

"""
Testing block for syntax sugar
"""

class Add(ScalerOpBlock):
    inplace = True
    def __init__(self, dtype, pre=True, inplace=False):
        super().__init__(dtype, pre)
        self.inplace = inplace
        
    def forward(self, inputs):
        if self.pre:
            return self.a + inputs
        else:
            if self.inplace:
                inputs += self.a
                return inputs
            else:
                return inputs + self.a
    
class Sub(ScalerOpBlock):
    inplace = True
    def __init__(self, dtype, pre=True, inplace=False):
        super().__init__(dtype, pre)
        self.inplace = inplace
        
    def forward(self, inputs):
        if self.pre:
            return self.a - inputs
        else:
            if self.inplace:
                inputs -= self.a
                return inputs
            else:
                return inputs - self.a
    
class Mul(ScalerOpBlock):
    inplace = True
    def __init__(self, dtype, pre=True, inplace=False):
        super().__init__(dtype, pre)
        self.inplace = inplace
        
    def forward(self, inputs):
        if self.pre:
            return self.a * inputs
        else:
            if self.inplace:
                inputs *= self.a
                return inputs
            else:
                return inputs * self.a
    
class Div(ScalerOpBlock):
    inplace = True
    def __init__(self, dtype, pre=True, inplace=False):
        super().__init__(dtype, pre)
        self.inplace = inplace
        
    def forward(self, inputs):
        if self.pre:
            return self.a / inputs
        else:
            if self.inplace:
                inputs /= self.a
                return inputs
            else:
                return inputs / self.a

class Pow(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        if self.pre:
            return self.a ** inputs
        else:
            return inputs ** self.a
    

"""
Testing block for torch.op
"""

class TorchAdd(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        return torch.add(self.a, inputs) if self.pre else torch.add(inputs, self.a)
    
class TorchSub(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        return torch.sub(self.a, inputs) if self.pre else torch.sub(inputs, self.a)
    
class TorchMul(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        return torch.mul(self.a, inputs) if self.pre else torch.mul(inputs, self.a)
    
class TorchDiv(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        return torch.div(self.a, inputs) if self.pre else torch.div(inputs, self.a)
    
class TorchTrueDiv(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        return torch.true_divide(self.a, inputs) if self.pre else torch.true_divide(inputs, self.a)

class TorchPow(ScalerOpBlock):
    def __init__(self, dtype, pre=True):
        super().__init__(dtype, pre)
        
    def forward(self, inputs):
        return torch.pow(self.a, inputs) if self.pre else torch.pow(inputs, self.a)

class UnitsModel(nn.Module):
    # TODO: add matmul block
    syntax_sugar_op_test = {
        "blocks": [Add, Mul, Sub, Div, Pow],
        "isscaler": True,
    }
    torch_op_test = {
        "blocks": [TorchAdd, TorchMul, TorchSub, TorchDiv, TorchTrueDiv, TorchPow],
        "isscaler": True,
    }
    # torch_tensor_op_test = [Add, Mul, Sub, Div, Pow]
    def __init__(self, test_name):
        super().__init__()
        self.test_name = test_name
        self.isscaler = getattr(self, self.test_name)["isscaler"]
        if self.isscaler:
            self.int_test = self.generate_block(int)
            self.float_test = self.generate_block(float)
        self.tensor_test = self.generate_block(torch.Tensor)
    
    def forward(self, inputs):
        if self.isscaler:
            inputs = self.int_test(inputs)
            inputs = self.float_test(inputs)
        inputs = self.tensor_test(inputs)
        return inputs
    
    def generate_block(self, dtype, noorder=False):
        blocks = []
        
        preblocks = []
        for block_func in getattr(self, self.test_name)["blocks"]:
            preblocks.append(block_func(dtype=dtype))
        tmp = nn.Sequential(*preblocks)
        tmp._get_name = lambda :"preblocks"
        blocks.append(tmp)
        
        inblocks = []    
        for block_func in getattr(self, self.test_name)["blocks"]:
            if block_func.inplace:
                inblocks.append(block_func(dtype=dtype, pre=False, inplace=True))
        tmp = nn.Sequential(*inblocks)
        tmp._get_name = lambda :"inblocks"
        blocks.append(tmp)
        
        if not noorder:
            postblocks = []
            for block_func in getattr(self, self.test_name)["blocks"]:
                postblocks.append(block_func(dtype=dtype, pre=False))
            tmp = nn.Sequential(*postblocks)
            tmp._get_name = lambda :"postblocks"
            blocks.append(tmp)
        
        return nn.Sequential(*blocks)