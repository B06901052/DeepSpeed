import time
import typing
import logging
import functools
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from flops_profiler.utils import *

old_functions = {}
try:
    import torchaudio
    old_functions.update({
        (torchaudio.compliance.kaldi, func): False for func in [
            "Tensor", "Tuple",
            "_next_power_of_2", "_get_waveform_and_window_properties", "_get_epsilon", "_get_strided",
            "inverse_mel_scale_scalar", "mel_scale_scalar",
        ]
    })
except ModuleNotFoundError:
    pass
try:
    import torchvision
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter('[%(module)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# print all functions called with their input/output (only when level == it will trigger)
logging.addLevelName(15, "SHOWFUNC")
logging.SHOWFUNC = 15 

logger.setLevel(logging.DEBUG)

Tensor = torch.Tensor

module_flop_count = []
module_mac_count = []
warning_msg = set()

# Here lists the functions which have not to been counted.
# torch.nn.functional
old_functions.update({
    (F, func): False for func in ["dropout", "dropout2d", "dropout3d", "split", "pad", "embedding"]
})
# torch._C._nn
old_functions.update({
    (torch._C._nn, func): False for func in [
        "pad_sequence",
        "reflection_pad1d", "reflection_pad2d", "reflection_pad3d",
        "replication_pad1d", "replication_pad2d", "replication_pad3d"
    ]
})
# torch.Tensor
old_functions.update({
    (torch.Tensor, func): False for func in [
    # bool
    "__and__", "__or__", "__xor__", "__lshift__", "__rshift__", "__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__",
    # other
    "__index__", "__getitem__", "__setitem__", "__invert__", "__format__", "__hash__", "contiguous",
    # comparison
    "__eq__", "__ge__", "__gt__", "__le__", "__lt__", "__ne__",
    "eq", "ge", "gt", "le", "lt", "ne", "all", "any", "min", "max",
    # assign
    "masked_fill", "masked_fill_", "fill", "fill_", "zero_",
    # info
    "__len__", "__format__", "__repr__",
    "dim", "shape", "size", "has_names", "data_ptr", "get_device", "numel", "is_floating_point",
    # device
    "to", "cpu", "cuda", "detach",
    # type conversion
    "__bool__", "__int__", "__float__",
    "type_as", "float", "double", "long", "bool", "tolist", "numpy", "item", "clone",
    # view or reshape
    "view", "expand", "repeat", "reshape", "transpose", "as_strided", "scatter", "scatter_", "chunk", "squeeze", "unsqueeze", "unbind", "permute", "split", "flip", "index_select", 
    ]
})
# torch
old_functions.update({
    (torch, func): False for func in [
        # comparison
        "eq", "ge", "gt", "le", "lt", "ne", "all", "any", "min", "max", "where", "isfinite", 
        "greater_equal", "greater", "less_equal", "less", "not_equal", 
        # normalization, see QA.md for more detail
        "batch_norm", "group_norm", "instance_norm", "layer_norm", 
        # shape-related
        "cat", "stack", "vstack", "hstack", "sstack", "row_stack", "column_stack", "slice", "chunk", "flip", 
        # creation
        "empty_like", "full_like", "ones_like", "randn_like", "zeros_like", 
        # info
        "numel", 
        # other
        "embedding", 
    ]
})


class FlopsProfiler(object):
    """Measures the latency, number of estimated floating-point operations and parameters of each module in a PyTorch model.

    The flops-profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how latency, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top k modules in terms of aggregated latency, flops, and parameters at depth l with k and l specified by the user. The output profile is computed for each batch of input.
    The DeepSpeed flops profiler can be used with the DeepSpeed runtime or as a standalone package.
    When using DeepSpeed for model training, the flops profiler can be configured in the deepspeed_config file and no user code change is required.

    If using the profiler as a standalone package, one imports the flops_profiler package and use the APIs.

    Here is an example for usage in a typical training workflow:

        .. code-block:: python

            model = Model()
            prof = FlopsProfiler(model)

            for step, batch in enumerate(data_loader):
                if step == profile_step:
                    prof.start_profile()

                loss = model(batch)

                if step == profile_step:
                    flops = prof.get_total_flops(as_string=True)
                    params = prof.get_total_params(as_string=True)
                    prof.print_model_profile(profile_step=profile_step)
                    prof.end_profile()

                loss.backward()
                optimizer.step()

    To profile a trained model in inference, use the `get_model_profile` API.

    Args:
        object (torch.nn.Module): The PyTorch model to profile.
    """
    def __init__(self, model, ds_engine=None, show_untracked=False, show_time=True, precision=2):
        self.model = model
        self.ds_engine = ds_engine
        self.started = False
        self.func_patched = False
        self.show_untracked = show_untracked
        self.show_time = show_time
        self.precision = precision
        if not self.show_untracked:
            def untracked_filter(record):
                return record.msg.find("Untracked function") < 0
            logger.addFilter(untracked_filter)

    def start_profile(self, ignore_list=None):
        """Starts profiling.

        Extra attributes are added recursively to all the modules and the profiled torch.nn.functionals are monkey patched.

        Args:
            ignore_list (list, optional): the list of modules to ignore while profiling. Defaults to None.
        """
        self.reset_profile()
        _patch_torch()
        try:
            import torchaudio
            _patch_torchaudio()
        except ModuleNotFoundError:
            pass
        try:
            import torchvision
            _patch_torchvision()
        except ModuleNotFoundError:
            pass

        def register_module_hooks(module, ignore_list):
            def pre_hook(module, input):
                module_flop_count.append([])
                module_mac_count.append([])

            module.__pre_hook_handle__ = module.register_forward_pre_hook(pre_hook)
                    
            # if computing the flops of a module directly
            if type(module) in MODULE_HOOK_MAPPING:
                def post_hook(module, input, output):
                    module_flop_count.pop()
                    module_mac_count.pop()
                    MODULE_HOOK_MAPPING[type(module)](module, input, output)
            # if ignore the module
            elif ignore_list and type(module) in ignore_list:
                def post_hook(module, input, output):
                    module_flop_count.pop()
                    module_mac_count.pop()
            # if computing the flops of the functionals in a module
            else:
                def post_hook(module, input, output):
                    module.__flops__ += sum(elem[1] for elem in module_flop_count.pop())
                    module.__macs__ += sum(elem[1] for elem in module_mac_count.pop())

            module.__post_hook_handle__ = module.register_forward_hook(post_hook)

            if self.show_time:
                def start_time_hook(module, input):
                    torch.cuda.synchronize()
                    module.__start_time__ = time.time()

                module.__start_time_hook_handle__ = module.register_forward_pre_hook(
                    start_time_hook)

                def end_time_hook(module, input, output):
                    torch.cuda.synchronize()
                    module.__duration__ += time.time() - module.__start_time__

                module.__end_time_hook_handle__ = module.register_forward_hook(end_time_hook)

        self.model.apply(functools.partial(register_module_hooks, ignore_list=ignore_list))
        self.started = True
        self.func_patched = True

    def stop_profile(self):
        """Stop profiling.

        All torch.nn.functionals are restored to their originals.
        """
        if self.started and self.func_patched:
            _reload_functions()
            self.func_patched = False

        def remove_profile_attrs(module):
            if isinstance(module, torch.jit.ScriptModule):
                logger.error("can't remove attr from SriptModule {}".format(module))
                return
            for name in ["pre_hook", "post_hook", "flops", "start_time_hook", "end_time_hook"]:
                name = f"__{name}_handle__"
                if hasattr(module, name):
                    getattr(module, name).remove()
                    delattr(module, name)

        self.model.apply(remove_profile_attrs)

    def reset_profile(self):
        """Resets the profiling.

        Adds or resets the extra attributes.
        """
        def add_or_reset_attrs(module):
            module.__flops__ = 0
            module.__macs__ = 0
            module.__params__ = sum(p.numel() for p in module.parameters())
            module.__start_time__ = 0
            module.__duration__ = 0

        self.model.apply(add_or_reset_attrs)

    def end_profile(self):
        """Ends profiling.

        The added attributes and handles are removed recursively on all the modules.
        """
        if not self.started:
            return
        for msg in sorted(warning_msg):
            logging.warning(msg)
        self.stop_profile()
        self.started = False

        def remove_profile_attrs(module):
            if isinstance(module, torch.jit.ScriptModule):
                logger.error("can't remove attr from SriptModule {}".format(module))
                return
            for name in ["flops", "macs", "params", "start_time", "duration"]:
                name = f"__{name}__"
                if hasattr(module, name):
                    delattr(module, name)

        self.model.apply(remove_profile_attrs)

    def get_total_flops(self, as_string=False):
        """Returns the total flops of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_flops = get_module_flops(self.model)
        return num_to_string(total_flops, precision=self.precision) if as_string else total_flops

    def get_total_macs(self, as_string=False):
        """Returns the total MACs of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_macs = get_module_macs(self.model)
        return macs_to_string(total_macs, precision=self.precision) if as_string else total_macs

    def get_total_duration(self, as_string=False):
        """Returns the total duration of the model forward pass.

        Args:
            as_string (bool, optional): whether to output the duration as string. Defaults to False.

        Returns:
            The latency of the model forward pass.
        """
        total_duration = get_module_duration(self.model)
        return duration_to_string(total_duration, precision=self.precision) if as_string else total_duration

    def get_total_params(self, as_string=False):
        """Returns the total parameters of the model.

        Args:
            as_string (bool, optional): whether to output the parameters as string. Defaults to False.

        Returns:
            The number of parameters in the model.
        """
        return params_to_string(
            self.model.__params__, precision=self.precision) if as_string else self.model.__params__

    def print_model_profile(self,
                            profile_step=1,
                            module_depth=-1,
                            top_modules=1,
                            detailed=True,
                            output_file=None,
                            device="cpu",
                            input_shape=None):
        """Prints the model graph with the measured profile attached to each module.

        Args:
            profile_step (int, optional): The global training step at which to profile. Note that warm up steps are needed for accurate time measurement.
            module_depth (int, optional): The depth of the model to which to print the aggregated module information. When set to -1, it prints information from the top to the innermost modules (the maximum depth).
            top_modules (int, optional): Limits the aggregated profile output to the number of top modules specified.
            detailed (bool, optional): Whether to print the detailed model profile.
            output_file (str, optional): Path to the output file. If None, the profiler prints to stdout.
        """
        if not self.started:
            return
        import sys
        import os.path
        original_stdout = None
        f = None
        if output_file and output_file != "":
            dir_path = os.path.dirname(output_file)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            original_stdout = sys.stdout
            f = open(output_file, "w")
            sys.stdout = f

        total_flops = self.get_total_flops()
        total_macs = self.get_total_macs()
        total_duration = self.get_total_duration()
        total_params = self.get_total_params()

        self.flops = total_flops
        self.macs = total_macs
        self.params = total_params

        print(
            "\n-------------------------- DeepSpeed Flops Profiler --------------------------"
        )
        print(f'Profile Summary at step {profile_step}:')
        print(
            "Notations:\ndata parallel size (dp_size), model parallel size(mp_size),\nnumber of parameters (params), number of multiply-accumulate operations(MACs),\nnumber of floating-point operations (FLOPs), floating-point operations per second (FLOPS),\nfwd latency (forward propagation latency), bwd latency (backward propagation latency),\nstep (weights update latency), iter latency (sum of fwd, bwd and step latency)\n"
        )
        if self.ds_engine:
            print('{:<60}  {:<8}'.format('world size: ', self.ds_engine.world_size))
            print('{:<60}  {:<8}'.format('data parallel size: ',
                                         self.ds_engine.dp_world_size))
            print('{:<60}  {:<8}'.format('model parallel size: ',
                                         self.ds_engine.mp_world_size))
            print('{:<60}  {:<8}'.format(
                'batch size per GPU: ',
                self.ds_engine.train_micro_batch_size_per_gpu()))

        print('{:<60}  {}'.format('device: ', device))
        print('{:<60}  {}'.format('input shape: ', input_shape))
        print('{:<60}  {:<8}'.format('params per gpu: ', params_to_string(total_params, precision=self.precision)))
        print('{:<60}  {:<8}'.format(
            'params of model = params per GPU * mp_size: ',
            params_to_string(total_params *
                             (self.ds_engine.mp_world_size) if self.ds_engine else 1, precision=self.precision)))

        print('{:<60}  {:<8}'.format('fwd MACs per GPU: ', macs_to_string(total_macs, precision=self.precision)))

        print('{:<60}  {:<8}'.format('fwd FLOPs per GPU: ', flops_to_string(total_flops, precision=self.precision)))

        print('{:<60}  {:<8}'.format(
            'fwd FLOPs of model = fwd FLOPs per GPU * mp_size: ',
            num_to_string(total_flops *
                          (self.ds_engine.mp_world_size) if self.ds_engine else 1, precision=self.precision)))

        fwd_latency = self.get_total_duration()
        if self.ds_engine and self.ds_engine.wall_clock_breakdown():
            fwd_latency = self.ds_engine.timers('forward').elapsed(False)
        if self.show_time:
            print('{:<60}  {:<8}'.format('fwd latency: ', duration_to_string(fwd_latency, precision=self.precision)))
            print('{:<60}  {:<8}'.format(
                'fwd FLOPS per GPU = fwd FLOPs per GPU / fwd latency: ',
                flops_to_string(total_flops / fwd_latency, precision=self.precision)))

        if self.ds_engine and self.ds_engine.wall_clock_breakdown():
            bwd_latency = self.ds_engine.timers('backward').elapsed(False)
            step_latency = self.ds_engine.timers('step').elapsed(False)
            if self.show_time:
                print('{:<60}  {:<8}'.format('bwd latency: ',
                                             duration_to_string(bwd_latency, precision=self.precision)))
                print('{:<60}  {:<8}'.format(
                    'bwd FLOPS per GPU = 2 * fwd FLOPs per GPU / bwd latency: ',
                    flops_to_string(2 * total_flops / bwd_latency, precision=self.precision)))
                print('{:<60}  {:<8}'.format(
                    'fwd+bwd FLOPS per GPU = 3 * fwd FLOPs per GPU / (fwd+bwd latency): ',
                    flops_to_string(3 * total_flops / (fwd_latency + bwd_latency), precision=self.precision)))

                print('{:<60}  {:<8}'.format('step latency: ',
                                             duration_to_string(step_latency, precision=self.precision)))

                iter_latency = fwd_latency + bwd_latency + step_latency
                print('{:<60}  {:<8}'.format('iter latency: ',
                                             duration_to_string(iter_latency, precision=self.precision)))
                print('{:<60}  {:<8}'.format(
                    'FLOPS per GPU = 3 * fwd FLOPs per GPU / iter latency: ',
                    flops_to_string(3 * total_flops / iter_latency, precision=self.precision)))

                samples_per_iter = self.ds_engine.train_micro_batch_size_per_gpu(
                ) * self.ds_engine.world_size
                print('{:<60}  {:<8.2f}'.format('samples/second: ',
                                                samples_per_iter / iter_latency))

        def flops_repr(module):
            params = module.__params__
            flops = get_module_flops(module)
            macs = get_module_macs(module)
            items = [
                params_to_string(params, precision=self.precision),
                "{:.2%} Params".format(params / (total_params + 1e-9)),
                macs_to_string(macs, precision=self.precision),
                "{:.2%} MACs".format(0.0 if total_macs == 0 else macs / (total_macs + 1e-9)),
            ]
            duration = get_module_duration(module)

            if self.show_time:
                items.append(duration_to_string(duration, precision=self.precision))
                items.append(
                    "{:.2%} latency".format(0.0 if total_duration == 0 else duration /
                                            total_duration))
            items.append(flops_to_string(flops, precision=self.precision))
            if self.show_time:
                items.append(flops_to_string(0.0 if duration == 0 else flops / duration, precision=self.precision))
            items.append(module.original_extra_repr())
            return ", ".join(items)

        def add_extra_repr(module):
            flops_extra_repr = flops_repr.__get__(module)
            if module.extra_repr != flops_extra_repr:
                module.original_extra_repr = module.extra_repr
                module.extra_repr = flops_extra_repr
                if module.extra_repr == module.original_extra_repr:
                    logger.error("{} failed to add_extra_repr".format(str(module)))

        def del_extra_repr(module):
            if hasattr(module, "original_extra_repr"):
                module.extra_repr = module.original_extra_repr
                try:
                    delattr(module, "original_extra_repr")
                except AttributeError:
                    logger.error("{} failed to del_extra_repr".format(str(module)))

        self.model.apply(add_extra_repr)

        print(
            "\n----------------------------- Aggregated Profile per GPU -----------------------------"
        )
        self.print_model_aggregated_profile(module_depth=module_depth,
                                            top_modules=top_modules)

        if detailed:
            print(
                "\n".join([
                    "\n------------------------------ Detailed Profile per GPU ------------------------------",
                    "Each module profile is listed after its name in the following order:",
                    "params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS\n",
                    "Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'."
                    "2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput."
                    "3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.\n"
                ])
            )
            print(self.model)

        self.model.apply(del_extra_repr)

        print("-" * 78)

        if output_file:
            sys.stdout = original_stdout
            f.close()

    def print_model_aggregated_profile(self, module_depth=-1, top_modules=1):
        """Prints the names of the top top_modules modules in terms of aggregated time, FLOPs, and parameters at depth module_depth.

        Args:
            module_depth (int, optional): the depth of the modules to show. Defaults to -1 (the innermost modules).
            top_modules (int, optional): the number of top modules to show. Defaults to 1.
        """
        info = {}
        if not hasattr(self.model, "__flops__"):
            print(
                "no __flops__ attribute in the model, call this function after start_profile and before end_profile"
            )
            return

        def walk_module(module, curr_depth, info):
            if curr_depth not in info:
                info[curr_depth] = {}
            if module.__class__.__name__ not in info[curr_depth]:
                info[curr_depth][module.__class__.__name__] = [0, 0, 0, 0]  # macs, params, time, flops
            info[curr_depth][module.__class__.__name__][0] += get_module_macs(module)
            info[curr_depth][module.__class__.__name__][1] += module.__params__
            info[curr_depth][module.__class__.__name__][2] += get_module_duration(module)
            info[curr_depth][module.__class__.__name__][3] += get_module_flops(module)
            has_children = len(module._modules.items()) != 0
            if has_children:
                for child in module.children():
                    walk_module(child, curr_depth + 1, info)

        walk_module(self.model, 0, info)

        depth = module_depth
        if module_depth == -1:
            depth = len(info) - 1

        print(
            f'Top {top_modules} modules in terms of params, MACs or fwd latency at different model depths:'
        )

        for d in range(depth):
            num_items = min(top_modules, len(info[d]))
            
            result = [None] * 4
            for i, func in enumerate([macs_to_string, params_to_string, duration_to_string, flops_to_string]):
                result[i] = {
                    k: func(v[i], precision=self.precision)
                    for k, v in sorted(info[d].items(),
                                key=lambda item: item[1][i],
                                reverse=True)[:num_items]
                }

            sort_macs, sort_params, sort_time, sort_flops = result

            print(f"depth {d}:")
            print(f"    params      - {sort_params}")
            print(f"    MACs        - {sort_macs}")
            print(f"    FLOPs       - {sort_flops}")
            if self.show_time:
                print(f"    fwd latency - {sort_time}")


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = input.numel() * out_features
    return 2 * macs, macs


def _xlu_flops_compute(input, *args, **kwargs):
    return input.numel(), 0


def _pool_flops_compute(input, *args, **kwargs):
    return input.numel(), 0


def _conv_flops_compute(input,
                        weight,
                        bias=None,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1):
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding, ) * length
    strides = stride if type(stride) is tuple else (stride, ) * length
    dilations = dilation if type(dilation) is tuple else (dilation, ) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] -
                      (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(output_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _conv_trans_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[-2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding, ) * length
    strides = stride if type(stride) is tuple else (stride, ) * length
    dilations = dilation if type(dilation) is tuple else (dilation, ) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):

        output_dim = (input_dim + 2 * paddings[idx] -
                      (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(input_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(_prod(output_dims))

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _batch_norm_flops_compute(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    factor = 1
    # has_affine
    if weight is not None:
        factor += 1
    # estimation
    if training:
        factor += 3
    return input.numel() * factor, 0


def _layer_norm_flops_compute(
    input: Tensor,
    normalized_shape: typing.List[int],
    weight: typing.Optional[Tensor] = None,
    bias: typing.Optional[Tensor] = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _group_norm_flops_compute(input: Tensor,
                              num_groups: int,
                              weight: typing.Optional[Tensor] = None,
                              bias: typing.Optional[Tensor] = None,
                              eps: float = 1e-5):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _instance_norm_flops_compute(
    input: Tensor,
    running_mean: typing.Optional[Tensor] = None,
    running_var: typing.Optional[Tensor] = None,
    weight: typing.Optional[Tensor] = None,
    bias: typing.Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4), 0


def _upsample_flops_compute(input,
                            size=None,
                            scale_factor=None,
                            mode="nearest",
                            align_corners=None):
    if isinstance(size, tuple):
        return int(_prod(size)), 0
    elif size:
        return int(size), 0
    assert scale_factor is not None, "either size or scale_factor should be defined"
    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == input.dim():
        flops *= int(_prod(scale_factor))
    else:
        flops *= scale_factor**input.dim()
    return flops, 0


def _softmax_flops_compute(input, *args, **kwargs):
    return input.numel() * 3, 0 # exp, sum, div


def _scatter_add_flops_compute(index, *args, **kwargs):
    result = index.numel()
    return result, result >> 1


def _multi_head_attention_forward_flops_compute(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: torch.Tensor,
        in_proj_bias: typing.Optional[torch.Tensor],
        bias_k: typing.Optional[torch.Tensor],
        bias_v: typing.Optional[torch.Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: torch.Tensor,
        out_proj_bias: typing.Optional[torch.Tensor],
        training: bool = True,
        key_padding_mask: typing.Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: typing.Optional[torch.Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: typing.Optional[torch.Tensor] = None,
        k_proj_weight: typing.Optional[torch.Tensor] = None,
        v_proj_weight: typing.Optional[torch.Tensor] = None,
        static_k: typing.Optional[torch.Tensor] = None,
        static_v: typing.Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        out: typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]] = None,
    ):
    """
    Count flops for torch.nn.functional.multi_head_attention_forward
    """
    assert query.dim() in [2, 3]
    assert query.dim() == key.dim()
    assert query.dim() == value.dim()

    if query.dim() == 3:
        batch_size = query.shape[1]
        assert key.shape[1] == batch_size
        assert value.shape[1] == batch_size
    else:
        batch_size = 1

    embed_dim = query.shape[-1]
    assert embed_dim == embed_dim_to_check
    assert embed_dim % num_heads == 0

    head_dim = embed_dim // num_heads
    tgt_len, src_len = query.shape[0], key.shape[0]

    if use_separate_proj_weight:
        assert key.shape[:-1] == value.shape[:-1]
    else:
        assert key.shape == value.shape

    flops, macs = 0, 0

    # flops and macs for in-projection.
    if not use_separate_proj_weight:
        # using in_proj_weight, which is of shape (3E, E), where E = embed_dim.
        n = query.numel() * embed_dim + 2 * key.numel() * embed_dim
        flops += n << 1
        macs += n
    else:
        n = (
            query.numel() * q_proj_weight.shape[0]
            + 2 * key.numel() * k_proj_weight.shape[0]
        )
        flops += n << 1
        macs += n

    if in_proj_bias is not None:
        n = query.numel() + key.numel() + value.numel()
        flops += n
        macs += n

    # q = q / sqrt(head_dim)
    flops += query.numel()
    macs += query.numel()

    # q * k^T (bmm)
    n = batch_size * num_heads * src_len * tgt_len * head_dim
    flops += n << 1
    macs += n

    # attn_mask
    n = batch_size * num_heads * src_len * tgt_len
    flops += n
    macs += n

    # softmax
    n = batch_size * num_heads * src_len * tgt_len
    flops += 3 * n
    macs += n << 1

    # dropout
    if dropout_p > 0.0:
        n = batch_size * num_heads * src_len * tgt_len
        flops += n
        macs += n

    # attn * v
    n = batch_size * num_heads * src_len * tgt_len * head_dim
    flops += n << 1
    macs += n

    # out-projection
    n = batch_size * tgt_len * embed_dim * out_proj_weight.shape[0]
    flops += n << 1
    macs += n

    return flops, macs


def _matmul_flops_compute(input, other, *, out=None):
    """
    Count flops for the matmul operation.
    """
    macs = _prod(input.shape) * other.shape[-1]
    return macs << 1, macs


def _addmm_flops_compute(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return (macs << 1) + _prod(input.shape), macs


def _einsum_flops_compute(equation, *operands):
    """
    Count flops for the einsum operation.
    """
    equation = equation.replace(" ", "")
    input_shapes = [o.shape for o in operands]

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
    for line in optim.split("\n"):
        if "optimized flop" in line.lower():
            flop = int(float(line.split(":")[-1]))
            return flop, flop>>1
    raise NotImplementedError("Unsupported einsum operation.")


def _tensor_addmm_flops_compute(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the tensor addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return (macs << 1) + _prod(self.shape), macs


def _elementwise_flops_compute(input, *args, **kwargs):
    other = kwargs.get("other", kwargs.get("exponent"))
    if args:
        other = args[0]
    assert other is not None, "_elementwise_flops_compute applies on a function in a wrong way"
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return _prod(other.shape), 0
        else:
            return 1, 0
    elif not torch.is_tensor(other):
        return _prod(input.shape), 0
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = _prod(final_shape)
        return flops, 0

def _unary_flops_compute_generator(flop_multiplier=1, flop_bias=0, macs_multiplier=1, macs_bias=0, has_correction=False):
    def _unary_flops_compute(input, *args, **kwargs):
        dim = kwargs.get("dim", -1)
        # var, std, var_mean
        biased = (
            kwargs.get("correction") is None and
            not kwargs.get("unbiased", False)
        )
        biased |= (
            kwargs.get("unbiased") is None and
            kwargs.get("correction") is not None and
            kwargs.get("correction", 0) > 0
        )
        biased &= has_correction
        
        op_num, other_num = 1, 1
        if dim == -1:
            op_num = _prod(input.shape)
             
        elif type(dim) == int:
            op_num = input.shape[dim]
            other_num = _prod(input.shape) // op_num
        
        elif type(dim) == tuple:
            for i in range(input.dim()):
                if i in dim:
                    op_num *= input.shape[i]
                else:
                    other_num *= input.shape[i]
            
        else:
            raise NotImplementedError

        return other_num * (flop_multiplier * op_num + flop_bias + biased), 0
    
    return _unary_flops_compute


def wrapFunc(module, func, name, funcFlopCompute):
    if old_functions.get((module, name), None) is False:
        return func
    
    elif old_functions.get((module, name), None) is not None:
        if func == old_functions[(module, name)]:
            return getattr(module, name)
        else:
            raise RuntimeError("keys are conflict in old_functions at {}.{}\nold func: {}\ncurrent func: {}".format(module, name, func, old_functions[(module, name)]))
    
    old_functions[(module, name)] = func

    @functools.wraps(func)
    def newFuncLogging(*args, **kwds):
        logger.log(logging.SHOWFUNC, module)
        logger.log(logging.SHOWFUNC, name + " is called !!!!")
        if isinstance(args[0], (torch.Tensor, int, float)):
            print("input:\n", args[0].detach().numpy())

        flops, macs = funcFlopCompute(*args, **kwds)
        logger.log(logging.SHOWFUNC, f"MACs: {macs}")
        
        module_flop_count[-1].append((name, flops))
        module_mac_count[-1].append((name, macs))
            
        flop_len = len(module_flop_count[-1])
        mac_len = len(module_mac_count[-1])
            
        result = func(*args, **kwds)
        
        # remove redundant count
        module_flop_count[-1] = module_flop_count[-1][:flop_len]
        module_mac_count[-1] = module_mac_count[-1][:mac_len]
        
        print("output:\n", result.detach().numpy())
        logger.log(logging.SHOWFUNC, name + " is done !!!!\n")      
        return result
    
    @functools.wraps(func)
    def newFunc(*args, **kwds):
        if module_flop_count:
            flop_len = len(module_flop_count[-1])
        if module_mac_count:
            mac_len = len(module_mac_count[-1])
        
        result = func(*args, **kwds)
        flops, macs = funcFlopCompute(*args, **kwds)
        
        # remove redundant count
        module_flop_count[-1] = module_flop_count[-1][:flop_len]
        module_flop_count[-1].append((name, flops))
        module_mac_count[-1] = module_mac_count[-1][:mac_len]
        module_mac_count[-1].append((name, macs))
        
        return result

    return newFuncLogging if logger.level == logging.SHOWFUNC else newFunc

def wrapWarning(module, func, name):
    if old_functions.get((module, name), None) is False:
        return func
    
    elif old_functions.get((module, name), None) is not None:
        if func == old_functions[(module, name)]:
            return getattr(module, name)
        else:
            raise RuntimeError("keys are conflict in old_functions at {}.{}\nold func: {}\ncurrent func: {}".format(module, name, func, old_functions[(module, name)]))
    old_functions[(module, name)] = func


    @functools.wraps(func)
    def newFunc(*args, **kwds):
        warning_msg.add("forward an unimplemented function: {}.{}".format(getattr(module, "__name__", module), name))

        flop_len = len(module_flop_count[-1])
        mac_len = len(module_mac_count[-1])
            
        result = func(*args, **kwds)
        
        # remove redundant count
        module_flop_count[-1] = module_flop_count[-1][:flop_len]
        module_mac_count[-1] = module_mac_count[-1][:mac_len]
        return result
    

    return newFunc

# for checking unimplemented part in function level
def _check_function_level_patch(pytorch_module):
    count = 0
    overridable_functions = torch.overrides.get_overridable_functions()[pytorch_module]
    for name in dir(pytorch_module):
        func = getattr(pytorch_module, name)
        if (
            (
                func in overridable_functions or # exclude func from typing
                not pytorch_module.__name__.startswith("torch.") or
                pytorch_module.__name__.startswith("torch._")
            ) and 
            hasattr(func, "__call__") and
            not (pytorch_module, name) in old_functions
        ):
            count += 1
            logger.info("[{}] Untracked function: {}".format(pytorch_module.__name__, name))
            setattr(pytorch_module, name, wrapWarning(pytorch_module, func, name))
    
    logger.info("[{}] Untracked function count: {}".format(pytorch_module.__name__, count))
    
def _check_operator_level_patch(pytorch_module):
    count = 0
    overridable_functions = torch.overrides.get_overridable_functions()[pytorch_module]
    for func, name in map(lambda x: (x, x.__name__), overridable_functions):
        if (
            hasattr(func, "__call__") and
            not (pytorch_module, name) in old_functions
        ):
            count += 1
            logger.info("[{}] Untracked function: {}".format(pytorch_module.__name__, name))
            setattr(pytorch_module, name, wrapWarning(pytorch_module, func, name))
    
    logger.info("[{}] Untracked function count: {}".format(pytorch_module.__name__, count))
 
def _patch_torch():
    # functional level
    _patch_fft()# 16
    _patch_nn_functionals()# 74
    _patch_linalg()# 26
    _patch_special()# 11
    
    # operator level
    _patch_tensor_methods()
    
def _patch_torchvision():
    # TODO: finish torchvision.{ops, transforms}, make it optional
    _check_function_level_patch(torchvision.ops)
    _check_function_level_patch(torchvision.transforms)
    pass
 
def _patch_torchaudio():# 42
    # TODO: finish torchaudio.functional, make it optional
    _check_function_level_patch(torchaudio.functional)
    _check_function_level_patch(torchaudio.compliance.kaldi)
    pass
    
def _patch_fft():
    # TODO: finish fft
    _check_function_level_patch(torch.fft)

def _patch_nn_functionals():
    # FC
    F.linear = wrapFunc(F, F.linear, "linear", _linear_flops_compute)
    torch._C._nn.linear = wrapFunc(torch._C._nn, torch._C._nn.linear, "linear", _linear_flops_compute)

    # convolutions
    F.conv1d = wrapFunc(F, F.conv1d, "conv1d", _conv_flops_compute)
    F.conv2d = wrapFunc(F, F.conv2d, "conv2d", _conv_flops_compute)
    F.conv3d = wrapFunc(F, F.conv3d, "conv3d", _conv_flops_compute)

    # conv transposed
    F.conv_transpose1d = wrapFunc(F, F.conv_transpose1d, "conv_transpose1d", _conv_trans_flops_compute)
    F.conv_transpose2d = wrapFunc(F, F.conv_transpose2d, "conv_transpose2d", _conv_trans_flops_compute)
    F.conv_transpose3d = wrapFunc(F, F.conv_transpose3d, "conv_transpose3d", _conv_trans_flops_compute)

    # activations
    for name in ["relu", "prelu", "elu", "leaky_relu", "relu6", "silu", "gelu"]:
        if hasattr(F, name):
            setattr(F, name, wrapFunc(F, getattr(F, name), name, _xlu_flops_compute))

    # Normalizations
    F.batch_norm = wrapFunc(F, F.batch_norm, "batch_norm", _batch_norm_flops_compute)
    F.layer_norm = wrapFunc(F, F.layer_norm, "layer_norm", _layer_norm_flops_compute)    
    F.instance_norm = wrapFunc(F, F.instance_norm, "instance_norm", _instance_norm_flops_compute)
    F.group_norm = wrapFunc(F, F.group_norm, "group_norm", _group_norm_flops_compute)

    # poolings
    pools = map(lambda x: "".join(x), itertools.product(
        ["", "adaptive_"], ["avg_", "max_"], ["pool1d", 'pool2d', 'pool3d']
    ))
    for pool in pools:
        setattr(F, pool, wrapFunc(F, getattr(F, pool), pool, _pool_flops_compute))
        if hasattr(torch._C._nn, pool):
            setattr(torch._C._nn, pool, wrapFunc(torch._C._nn, getattr(torch._C._nn, pool), pool, _pool_flops_compute))

    # upsample
    F.upsample = wrapFunc(F, F.upsample, "upsample", _upsample_flops_compute)
    F.interpolate = wrapFunc(F, F.interpolate, "interpolate", _upsample_flops_compute)

    # softmax
    F.softmax = wrapFunc(F, F.softmax, "softmax", _softmax_flops_compute)

    # multi_head_attention_forward
    F.multi_head_attention_forward = wrapFunc(
        F,
        F.multi_head_attention_forward,
        "multi_head_attention_forward",
        _multi_head_attention_forward_flops_compute,
    )
    
    # not implemented
    _check_function_level_patch(F)
    _check_function_level_patch(torch._C._nn)

def _patch_linalg():
    # TODO: finish linalg
    _check_function_level_patch(torch.linalg)
    
def _patch_special():
    # TODO: finish special
    _check_function_level_patch(torch.special)

def _patch_tensor_methods():
    torch.matmul = wrapFunc(torch, torch.matmul, "matmul", _matmul_flops_compute)
    torch.Tensor.matmul = wrapFunc(torch.Tensor, torch.Tensor.matmul, "matmul", _matmul_flops_compute)
    torch.Tensor.__matmul__ = wrapFunc(torch.Tensor, torch.Tensor.__matmul__, "__matmul__", _matmul_flops_compute)
    
    torch.mm = wrapFunc(torch, torch.mm, "mm", _matmul_flops_compute)
    torch.Tensor.mm = wrapFunc(torch.Tensor, torch.Tensor.mm, "mm", _matmul_flops_compute)
    torch.bmm = wrapFunc(torch, torch.bmm, "bmm", _matmul_flops_compute)
    torch.Tensor.bmm = wrapFunc(torch.Tensor, torch.Tensor.bmm, "bmm", _matmul_flops_compute)

    torch.addmm = wrapFunc(torch, torch.addmm, "addmm", _addmm_flops_compute)
    torch.Tensor.addmm = wrapFunc(torch.Tensor, torch.Tensor.addmm, "addmm", _tensor_addmm_flops_compute)
    
    torch.softmax = wrapFunc(torch, torch.softmax, "softmax", _softmax_flops_compute)
    torch.Tensor.softmax = wrapFunc(torch.Tensor, torch.Tensor.softmax, "softmax", _softmax_flops_compute)
    
    # activations
    torch.relu = wrapFunc(torch, torch.relu, "relu", _xlu_flops_compute)
    torch.Tensor.relu = wrapFunc(torch.Tensor, torch.Tensor.relu, "relu", _xlu_flops_compute)
    torch.prelu = wrapFunc(torch, torch.prelu, "prelu", _xlu_flops_compute)
    
    # others
    torch.scatter_add = wrapFunc(torch, torch.scatter_add, "scatter_add", _scatter_add_flops_compute)
    torch.Tensor.scatter_add = wrapFunc(torch.Tensor, torch.Tensor.scatter_add, "scatter_add", _scatter_add_flops_compute)
    torch.Tensor.scatter_add_ = wrapFunc(torch.Tensor, torch.Tensor.scatter_add_, "scatter_add_", _scatter_add_flops_compute)
    
    for op in ["add", "sub", "mul", "truediv", "floordiv", "div", "pow"]:
        # syntax sugar
        sugar_op = "__{}__".format(op)
        setattr(torch.Tensor, sugar_op, wrapFunc(torch.Tensor, getattr(torch.Tensor, sugar_op), sugar_op, _elementwise_flops_compute))
        
        # inplace syntax sugar
        sugar_op = "__i{}__".format(op)
        setattr(torch.Tensor, sugar_op, wrapFunc(torch.Tensor, getattr(torch.Tensor, sugar_op), sugar_op, _elementwise_flops_compute))
        # raw syntax sugar
        sugar_op = "__r{}__".format(op)
        setattr(torch.Tensor, sugar_op, wrapFunc(torch.Tensor, getattr(torch.Tensor, sugar_op), sugar_op, _elementwise_flops_compute))

        # torch.op and torch.Tensor.op     
        op = "true_divide" if op == "truediv" else op
        op = "floor_divide" if op == "floordiv" else op
        setattr(torch, op, wrapFunc(torch, getattr(torch, op), op, _elementwise_flops_compute))
        setattr(torch.Tensor, op, wrapFunc(torch.Tensor, getattr(torch.Tensor, op), op, _elementwise_flops_compute))

    # alias
    for op in ["subtract", "multiply", "divide"]:
        setattr(torch, op, wrapFunc(torch, getattr(torch, op), op, _elementwise_flops_compute))
        setattr(torch.Tensor, op, wrapFunc(torch.Tensor, getattr(torch.Tensor, op), op, _elementwise_flops_compute))
        
    # https://pytorch.org/docs/stable/torch.html#math-operations
    math_ops = {
        # Pointwise Ops (https://pytorch.org/docs/stable/torch.html#pointwise-ops)
        "abs": {},
        "absolute": {},
        "acos": {},
        "arccos": {},
        "acosh": {},
        "arccosh": {},
        # add (implemented above)
        # addcdiv (not implemented)
        # addcmul (not implemented)
        # angle (not for floating-point)
        "asin": {},
        "arcsin": {},
        "asinh": {},
        "arcsinh": {},
        "atan": {},
        "arctan": {},
        "atanh": {},
        "arctanh": {},
        # atan2 (not implemented)
        # arctan2 (alias)
        # bitwise_{not, and, or, xor, left_shift, right_shift} (not for floating-point)
        "ceil": {},
        "clamp": {"flop_multiplier": 2},
        "clip": {"flop_multiplier": 2},
        # conj_physical (not for floating-point)
        "copysign": {},
        "cos": {},
        "cosh": {},
        "deg2rad": {},
        # div (implemented above)
        # divide (alias)
        # digamma (torch.special)
        # erf (torch.special)
        # erfc (torch.special)
        # erfinv (torch.special)
        "exp": {},
        # exp2 (torch.special)
        # expm1 (torch.special)
        # fake_quantize_per_channel_affine (not implemented)
        # fake_quantize_per_tensor_affine (not implemented)
        # fix (alias)
        # float_power (not implemented)
        "floor": {},
        # floor_divide (deprecated)
        # fmod (not implemented)
        "frac": {"flop_multiplier": 4}, # sub, abs, floor, sgn
        # TODO: check remaining math operations below
        "lgamma": {"flop_multiplier": 3}, # ln, gamma, abs
        "log": {},
        "log10": {},
        "log1p": {},
        "log2": {},
        "reciprocal": {},
        "round": {},
        "rsqrt": {"flop_multiplier": 2}, # sqrt, reciprocal
        "sigmoid": {"flop_multiplier": 2}, # add, exp, sub, reciprocal
        "sin": {},
        "sinc": {},
        "sinh": {},
        "sqrt": {},
        "square": {},
        "tan": {},
        "tanh": {},
        "trunc": {"flop_multiplier": 3}, # abs, floor, sgn
        # Reduction Ops (https://pytorch.org/docs/stable/torch.html#reduction-ops)
        "mean": {"flop_multiplier": 1, "flop_bias": 0},
        "sum": {"flop_multiplier": 1, "flop_bias": -1},
        "cumsum": {"flop_multiplier": 1, "flop_bias": -1},
        "cumsum_": {"flop_multiplier": 1, "flop_bias": -1},
        "var": {"flop_multiplier": 4, "flop_bias": 0, "has_correction": True},# mean(N), sub(N), pow(N), sum(N-1), avg(1)
        "var_mean": {"flop_multiplier": 4, "flop_bias": 0, "has_correction": True},
        "std": {"flop_multiplier": 4, "flop_bias": 1, "has_correction": True},
        "std_mean": {"flop_multiplier": 4, "flop_bias": 1, "has_correction": True},
    }
    
    not_in_tensor = {"var_mean", "std_mean"}
    
    for op in math_ops:
        funcFlopCompute = _unary_flops_compute_generator(
            flop_multiplier=math_ops[op].get("flop_multiplier", 1),
            flop_bias=math_ops[op].get("flop_bias", 0),
            has_correction=math_ops[op].get("has_correction", False)
        )
        # torch.op
        if hasattr(torch, op):
            setattr(torch, op, wrapFunc(torch, getattr(torch, op), op, funcFlopCompute))
        # torch.Tensor.op
        if op in not_in_tensor:
            continue
        setattr(torch.Tensor, op, wrapFunc(torch.Tensor, getattr(torch.Tensor, op), op, funcFlopCompute))
        
    
    torch.einsum = wrapFunc(torch, torch.einsum, "einsum", _einsum_flops_compute)
    
    _check_operator_level_patch(torch)
    _check_operator_level_patch(torch.Tensor)

def _reload_functions():
    for (module, name) in old_functions:
        old_func = old_functions[(module, name)]
        if old_func is not None and not isinstance(old_func, bool):
            setattr(module, name, old_func)
            old_functions[(module, name)] = None
            

def _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size * 3
        # last two hadamard _product and add
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size * 4
        # two hadamard _product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def _rnn_forward_hook(rnn_module, input, output):
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]

    if isinstance(inp, torch.nn.utils.rnn.PackedSequence):
        batch_size_mul_seq_length = inp.batch_sizes.sum()
    else:
        batch_size = inp.shape[0]
        seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__("weight_ih_l" + str(i))
        w_hh = rnn_module.__getattr__("weight_hh_l" + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__("bias_ih_l" + str(i))
            b_hh = rnn_module.__getattr__("bias_hh_l" + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    if isinstance(inp, torch.nn.utils.rnn.PackedSequence):
        flops *= batch_size_mul_seq_length
    else:
        flops *= batch_size
        flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)
    rnn_module.__macs__ += rnn_module.__flops__ >> 1


def _rnn_cell_forward_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = inp.shape[1]
    flops = _rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)
    rnn_cell_module.__macs__ += int(flops) >> 1


# RNN
MODULE_HOOK_MAPPING = {
    m: _rnn_cell_forward_hook for m in [nn.RNNCell, nn.LSTMCell, nn.GRUCell]
}
MODULE_HOOK_MAPPING.update({
    m: _rnn_forward_hook for m in [nn.RNN, nn.GRU, nn.LSTM]
})


def get_model_profile(
    model,
    input_shape=None,
    args=[],
    kwargs={},
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=1,
    warm_up=1,
    as_string=True,
    output_file=None,
    ignore_modules=None,
    show_untracked=False,
    show_time=True,
    precision=2,
):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Example:

    .. code-block:: python

        model = torchvision.models.alexnet()
        batch_size = 256
        flops, macs, params = get_model_profile(model=model, input_shape=(batch_size, 3, 224, 224)))

    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        input_shape (tuple): input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args (list): list of positional arguments to the model.
        kwargs (dict): dictionary of keyword arguments to the model.
        print_profile (bool, optional): whether to print the model profile. Defaults to True.
        detailed (bool, optional): whether to print the detailed model profile. Defaults to True.
        module_depth (int, optional): the depth into the nested modules. Defaults to -1 (the inner most modules).
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        warm_up (int, optional): the number of warm-up steps before measuring the latency of each module. Defaults to 1.
        as_string (bool, optional): whether to print the output as string. Defaults to True.
        output_file (str, optional): path to the output file. If None, the profiler prints to stdout.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    assert isinstance(model, nn.Module), "model must be a PyTorch module"
    prof = FlopsProfiler(model, show_untracked=show_untracked, show_time=show_time, precision=precision)
    model.eval()

    if input_shape is not None:
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"
        try:
            input = torch.ones(()).new_empty(
                (*input_shape,
                 ),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
        except StopIteration:
            input = torch.ones(()).new_empty((*input_shape, ))

        args = [input, *args]
    else:
        if isinstance(args[0], (torch.Tensor, np.ndarray)):
            input_shape = args[0].shape
        elif isinstance(args[0], (tuple, list)):
            input_shape = [i.shape for i in args[0]]
        else:
            raise RuntimeWarning("Fail to retrieve input shape.")

    assert (len(args) > 0) or (len(kwargs) > 0), "args and/or kwargs must be specified if input_shape is None"

    for _ in range(warm_up):
        _ = model(*args, **kwargs)

    prof.start_profile(ignore_list=ignore_modules)

    _ = model(*args, **kwargs)

    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(profile_step=warm_up,
                                 module_depth=module_depth,
                                 top_modules=top_modules,
                                 detailed=detailed,
                                 output_file=output_file,
                                 device=args[0][0].device,
                                 input_shape=input_shape)

    prof.end_profile()
    if as_string:
        return number_to_string(flops, precision=precision), macs_to_string(macs, precision=precision), params_to_string(params, precision=precision)

    return flops, macs, params
