"""
This file is models for unit test on oprations between scaler and tensor.
# Something special
* syntax_sugar_op_test
  * __rdiv__ will call __mul__ in it, the op count will twice. (not a bug)
  * __rpow__ and __pow__ both call torch.pow, so just count op in torch.pow
"""
import torch
import torch.nn as nn


class ScalerOpBlock(nn.Module):
    inplace=False
    def __init__(self, dtype, order):
        """ScalerOpBlock

        Args:
            dtype (type): The type of scaler
            order (str): The order of operation

        Raises:
            NotImplementedError: Raising if the dtype is not supported 
        """
        super().__init__()
        if dtype == int:
            self.a = 10
        elif dtype == float:
            self.a = 10.0
        elif dtype == torch.Tensor:
            self.a = torch.Tensor([10])
        elif dtype == nn.Parameter:
            self.a = nn.Parameter(torch.Tensor([10]))
        else:
            raise NotImplementedError
        
        self.order = order
        
    @classmethod
    def check_args_valid(cls, dtype, order):
        return not (
            order == "inplace" and not cls.inplace
        )

"""
test_name: syntax_sugar_scalerop_test
operation: +, -, *, /, **
Testing block for syntax sugar ops which interact with scaler
"""

class Add(ScalerOpBlock):
    inplace = True
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return self.a + inputs
        elif self.order == "post":
            return inputs + self.a
        elif self.order == "inplace":
            inputs += self.a
            return inputs
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
        
class Sub(ScalerOpBlock):
    inplace = True
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return self.a - inputs
        elif self.order == "post":
            return inputs - self.a
        elif self.order == "inplace":
            inputs -= self.a
            return inputs
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class Mul(ScalerOpBlock):
    inplace = True
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return self.a * inputs
        elif self.order == "post":
            return inputs * self.a
        elif self.order == "inplace":
            inputs *= self.a
            return inputs
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class Div(ScalerOpBlock):
    inplace = True
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return self.a / inputs
        elif self.order == "post":
            return inputs / self.a
        elif self.order == "inplace":
            inputs /= self.a
            return inputs
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))

class Pow(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return self.a + inputs
        elif self.order == "post":
            return inputs + self.a
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    

"""
test_name: torch_scalerop_test
operation: torch.add, torch.mul, torch.sub, torch.div, torch.true_divide, torch.pow
Testing block for torch ops which interact with scaler
"""

class TorchAdd(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.add(self.a, inputs)
        elif self.order == "post":
            return torch.add(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchSub(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.sub(self.a, inputs)
        elif self.order == "post":
            return torch.sub(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchMul(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.mul(self.a, inputs)
        elif self.order == "post":
            return torch.mul(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchDiv(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.div(self.a, inputs)
        elif self.order == "post":
            return torch.div(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchTrueDiv(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.true_divide(self.a, inputs)
        elif self.order == "post":
            return torch.true_divide(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))

class TorchPow(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.pow(self.a, inputs)
        elif self.order == "post":
            return torch.pow(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))

"""
test_name: torch_tensor_scalerop_test
operation: torch.Tensorl.add, torch.Tensorl.mul, torch.Tensorl.sub, torch.Tensorl.div, torch.Tensorl.true_divide, torch.Tensorl.pow
Testing block for torch.Tensor ops which interact with scaler
"""

class ScalerTensorOpBlock(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
    
    @classmethod
    def check_args_valid(cls, dtype, order):
        return not (
            dtype in {int, float} and order == "pre"
        ) and super().check_args_valid(dtype, order)

class TorchTensorAdd(ScalerTensorOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.add(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.add(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchTensorSub(ScalerTensorOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.sub(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.sub(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchTensorMul(ScalerTensorOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.mul(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.mul(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchTensorDiv(ScalerTensorOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.div(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.div(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchTensorTrueDiv(ScalerTensorOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.true_divide(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.true_divide(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))

class TorchTensorPow(ScalerTensorOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.pow(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.pow(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
        
syntax_sugar_scalerop_test = {
    "test_name": "syntax_sugar_scalerop_test",
    "modules": [Add, Mul, Sub, Div, Pow],
    "input_shape": (2, 2),
    "args": {
            "order": ["pre", "inplace", "post"],    
            "dtype": [int, float, torch.Tensor, nn.Parameter],    
    },
}

torch_scalerop_test = {
    "test_name": "torch_scalerop_test",
    "modules": [TorchAdd, TorchMul, TorchSub, TorchDiv, TorchTrueDiv, TorchPow],
    "input_shape": (2, 2),
    "args": {
            "order": ["pre", "post"],    
            "dtype": [int, float, torch.Tensor, nn.Parameter],    
    },
}

torch_tensor_scalerop_test = {
    "test_name": "torch_tensor_scalerop_test",
    "modules": [TorchTensorAdd, TorchTensorMul, TorchTensorSub, TorchTensorDiv, TorchTensorTrueDiv, TorchTensorPow],
    "input_shape": (2, 2),
    "args": {
            "order": ["pre", "post"],    
            "dtype": [int, float, torch.Tensor, nn.Parameter],    
    },
}