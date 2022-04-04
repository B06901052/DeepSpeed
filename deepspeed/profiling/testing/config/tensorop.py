"""
This file is models for unit test on oprations between tensors.
"""
import torch
import torch.nn as nn


class TensorOpBlock(nn.Module):
    inplace = False
    def __init__(self, weight_shape, wtype, dtype, order):
        super().__init__()
        
        self.order = order
        self.dtype = dtype
        self.wtype = wtype
        self.a = torch.empty((*weight_shape,), dtype=dtype)
        if self.wtype == nn.Parameter:
            self.a = nn.Parameter(self.a)
        elif self.wtype == torch.Tensor:
            pass
        else:
            raise NotImplementedError
        
    @classmethod
    def check_args_valid(cls, input_shape, weight_shape, wtype, dtype, order):
        return not (
            (order == "inplace" and not cls.inplace) or
            (input_shape)# FIXME: not done
        )
        

class Matmul(TensorOpBlock):
    inplace = True
    def __init__(self, weight_shape, wtype, dtype, order):
        super().__init__(weight_shape, wtype, dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return self.a @ inputs
        elif self.order == "post":
            return inputs @ self.a
        elif self.order == "inplace":
            inputs @= self.a
            return inputs
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))  


class TorchMatmul(TensorOpBlock):
    def __init__(self, weight_shape, wtype, dtype, order):
        super().__init__(weight_shape, wtype, dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.matmul(self.a, inputs)
        elif self.order == "post":
            return torch.matmul(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))


class TorchTensorMatmul(TensorOpBlock):
    def __init__(self, weight_shape, wtype, dtype, order):
        super().__init__(weight_shape, wtype, dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.matmul(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.matmul(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))

tensorop_test = {
    "test_name": "tensorop_test",
    "modules": [Matmul, TorchMatmul, TorchTensorMatmul],
    "input_shape": (4,4),
    "args": {
            "order": ["pre", "inplace", "post"],    
            "wtype": [torch.Tensor, nn.Parameter],
            "dtype": [torch.float],
            "weight_shape": [(4,4), (4,1), (1,4)]
    },
}