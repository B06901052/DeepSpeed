import numpy as np

"""
string formatter
"""
def _num_to_string(num, units=None, precision=2, unit="", order_names=["", "K", "M", "G", "T"], bias=0):
    order = int(min(max(np.log10(num)/3 + bias if num else 0, 0), len(order_names)-1))
        
    num *= 1000**(bias-order)
    if units is None:
        units = order_names[order] + unit
    
    return str(round(num, precision)) + " " + units

def num_to_string(num, precision=2):
    return _num_to_string(num, precision=precision, order_names=["", "K", "M", "G"])


def macs_to_string(macs, units=None, precision=2):
    return _num_to_string(macs, units=units, precision=precision, unit="MACs")


def number_to_string(num, units=None, precision=2):
    return _num_to_string(num, units=units, precision=precision)


def flops_to_string(flops, units=None, precision=2):
    return _num_to_string(flops, units=units, precision=precision, unit="FLOPs")


def params_to_string(params_num, units=None, precision=2):
    return _num_to_string(params_num, units=units, precision=precision, order_names=["", "k", "M", "G"])


def duration_to_string(duration, units=None, precision=2):
    return _num_to_string(duration, units=units, precision=precision, order_names=["n", "u", "m", ""], bias=3, unit="s")

"""
accumulate
"""
# can not iterate over all submodules using self.model.modules()
# since modules() returns duplicate modules only once
def get_module_flops(module):
    # iterate over immediate children modules
    return module.__flops__ + sum(map(get_module_flops, module.children()))


def get_module_macs(module):
    # iterate over immediate children modules
    return module.__macs__ + sum(map(get_module_macs, module.children()))


def get_module_duration(module):
    if module.__duration__ == 0:  # e.g. ModuleList
        return sum(map(lambda x: x.__duration__, module.children()))
    else:
        return module.__duration__