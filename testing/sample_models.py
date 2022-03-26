"""
This file is models for unit test.
"""
import inspect
import itertools

import torch
import torch.nn as nn

# %% [markdown]
# ## Something special
# * syntax_sugar_op_test
#   * __rdiv__ will call __mul__ in it, the op count will twice. (not a bug)
#   * __rpow__ and __pow__ both call torch.pow, so just count op in torch.pow

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

class TorchTensorAdd(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.add(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.add(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchTensorSub(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.sub(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.sub(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchTensorMul(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.mul(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.mul(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchTensorDiv(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.div(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.div(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))
    
class TorchTensorTrueDiv(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.true_divide(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.true_divide(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))

class TorchTensorPow(ScalerOpBlock):
    def __init__(self, dtype, order):
        super().__init__(dtype, order)
        
    def forward(self, inputs):
        if self.order == "pre":
            return torch.Tensor.pow(self.a, inputs)
        elif self.order == "post":
            return torch.Tensor.pow(inputs, self.a)
        else:
            raise NotImplementedError("undefined order {} for {}".format(self.order, self))


class UnitsModel(nn.Module):
    # TODO (joseph): add torch_tensor_op_test
    # TODO (joseph): add matmul block
    syntax_sugar_scalerop_test = {
        "test_name": "syntax_sugar_scalerop_test",
        "modules": [Add, Mul, Sub, Div, Pow],
        "args": {
                "order": ["pre", "inplace", "post"],    
                "dtype": [int, float, torch.Tensor],    
        },
    }
    torch_scalerop_test = {
        "test_name": "torch_scalerop_test",
        "modules": [TorchAdd, TorchMul, TorchSub, TorchDiv, TorchTrueDiv, TorchPow],
        "args": {
                "order": ["pre", "post"],    
                "dtype": [int, float, torch.Tensor],    
        },
    },
    torch_tensor_scalerop_test = {
        "test_name": "torch_tensor_scalerop_test",
        "modules": [TorchAdd, TorchMul, TorchSub, TorchDiv, TorchTrueDiv, TorchPow],
        "args": {
                "order": ["pre", "inplace", "post"],    
                "dtype": [int, float, torch.Tensor],    
        },
    },
    

    # torch_tensor_op_test = [Add, Mul, Sub, Div, Pow]
    def __init__(self, test_name="", test_config=None):
        """UnitsModel

        Args:
            test_name (str, optional): Assign an existing test defined in UnitsModel. Defaults to "".
            test_config (dict, optional): Customized an test, Defaults to None. It should include:
                test_name (str): The name of your test.
                modules (List[nn.Module]): The modules you want test
                args (Dict[str, list], optional): It should be:
                    keys (str): The name of arg
                    value (list): The values of arg
        """        
        super().__init__()
        if test_config:
            self.test_config = test_config
        else:
            self.test_config = inspect.getattr_static(self, test_name)
        
        self.net = self.generate_blocks()
    
    def forward(self, inputs):
        return self.net(inputs)
    
    def generate_blocks(self):
        nn.Sequential._get_name = lambda self: self.__name__
        allblocks = []
        for module in self.test_config["modules"]:
            module._get_name = lambda self: self.__name__
            blocks = []
            for args_cmb in itertools.product(*self.test_config["args"].values()):
                args_cmb = dict(zip(self.test_config["args"].keys(), args_cmb))
                if hasattr(module, "check_args_valid"):
                    if not module.check_args_valid(**args_cmb):
                        continue
                blocks.append(module(**args_cmb))
                blocks[-1].__name__ = "_".join(map(lambda x: str(x), args_cmb.values()))
            
            blocks = nn.Sequential(*blocks)
            blocks.__name__ = module.__name__ + "TestBlock"
            allblocks.append(blocks)
        
        allblocks = nn.Sequential(*allblocks)
        allblocks.__name__ = self.test_config["test_name"]
        
        return allblocks