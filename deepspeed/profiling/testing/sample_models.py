"""
This file is models for unit test.
"""
import inspect
import itertools

import torch
import torch.nn as nn

from config.scalerop import syntax_sugar_scalerop_test, torch_scalerop_test, torch_tensor_scalerop_test
from config.tensorop import tensorop_test
from config.unaryop import torch_unaryop_test

all_variables = dir()

"""
test_name: basic_tesnorop_test
operation: matmul, @, mm, bmm

"""

def list_all_defined_test():
    for v in all_variables:
        if v.endswith("_test"):
            print(v)

class UnitsModel(nn.Module):
    # TODO (joseph): add matmul block
    def __init__(self, test_name="", test_config=None):
        """UnitsModel

        Args:
            test_name (str, optional): Assign an existing test defined in UnitsModel. Defaults to "".
            test_config (dict, optional): Customized an test, Defaults to None. It should include:
                test_name (str): The name of your test.
                modules (List[nn.Module]): The modules you want test
                input_shape (Tuple[int]):
                args (Dict[str, list], optional): It should be:
                    keys (str): The name of arg
                    value (list): The values of arg
        """        
        super().__init__()
        if test_config:
            self.test_config = test_config
        elif test_name in all_variables:
            self.test_config = eval(test_name)
        else:
            print(dir())
            print(test_name)
            raise RuntimeError("You have to provide a test_name or customize your own test_config!")
        
        self.net = self.generate_blocks()
    
    def forward(self, inputs):
        return self.net(inputs)
    
    def generate_blocks(self):
        nn.Sequential._get_name = lambda self: self.__name__
        allblocks = []
        
        # iterate all modules
        for module in self.test_config["modules"]:
            module._get_name = lambda self: self.__name__
            blocks = []
            
            # iterate all combinations of args
            for args_cmb in itertools.product(*self.test_config["args"].values()):
                args_cmb = dict(zip(self.test_config["args"].keys(), args_cmb))
                # check the arg combination is valid
                if hasattr(module, "check_args_valid"):
                    if not module.check_args_valid(input_shape=self.test_config["input_shape"], **args_cmb):
                        continue
                blocks.append(module(**args_cmb))
                blocks[-1].__name__ = "_".join(map(lambda x: str(x), args_cmb.values()))
            
            # Merge to module level
            blocks = nn.Sequential(*blocks)
            blocks.__name__ = module.__name__ + "TestBlock"
            allblocks.append(blocks)
        
        # Merge to test level
        allblocks = nn.Sequential(*allblocks)
        allblocks.__name__ = self.test_config["test_name"]
        
        return allblocks