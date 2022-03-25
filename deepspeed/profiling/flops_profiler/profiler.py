import time
import typing
import logging
import inspect
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter('[%(module)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

Tensor = torch.Tensor

module_flop_count = []
module_mac_count = []

# FIXME: should consider two funcions from diff. modules while their names are the same.
# Here lists the functions which have not to been counted.
old_functions = {
    ("torch.nn.functional", "dropout"): F.dropout,
}


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
    def __init__(self, model, ds_engine=None):
        self.model = model
        self.ds_engine = ds_engine
        self.started = False
        self.func_patched = False

    def start_profile(self, ignore_list=None):
        """Starts profiling.

        Extra attributes are added recursively to all the modules and the profiled torch.nn.functionals are monkey patched.

        Args:
            ignore_list (list, optional): the list of modules to ignore while profiling. Defaults to None.
        """
        self.reset_profile()
        _patch_torch()
        _patch_torchaudio()

        def register_module_hooks(module, ignore_list):
            if ignore_list and type(module) in ignore_list:
                return

            # if computing the flops of a module directly
            if type(module) in MODULE_HOOK_MAPPING:
                module.__flops_handle__ = module.register_forward_hook(
                    MODULE_HOOK_MAPPING[type(module)])
                return

            # if computing the flops of the functionals in a module
            def pre_hook(module, input):
                module_flop_count.append([])
                module_mac_count.append([])

            module.__pre_hook_handle__ = module.register_forward_pre_hook(pre_hook)

            def post_hook(module, input, output):
                if module_flop_count:
                    module.__flops__ += sum([elem[1] for elem in module_flop_count[-1]])
                    module_flop_count.pop()
                    module.__macs__ += sum([elem[1] for elem in module_mac_count[-1]])
                    module_mac_count.pop()

            module.__post_hook_handle__ = module.register_forward_hook(post_hook)

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
            _reload_functionals()
            _reload_tensor_methods()
            self.func_patched = False

        def remove_profile_attrs(module):
            if hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__.remove()
                del module.__pre_hook_handle__
            if hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__.remove()
                del module.__post_hook_handle__
            if hasattr(module, "__flops_handle__"):
                module.__flops_handle__.remove()
                del module.__flops_handle__
            if hasattr(module, "__start_time_hook_handle__"):
                module.__start_time_hook_handle__.remove()
                del module.__start_time_hook_handle__
            if hasattr(module, "__end_time_hook_handle__"):
                module.__end_time_hook_handle__.remove()
                del module.__end_time_hook_handle__

        self.model.apply(remove_profile_attrs)

    def reset_profile(self):
        """Resets the profiling.

        Adds or resets the extra attributes.
        """
        def add_or_reset_attrs(module):
            module.__flops__ = 0
            module.__macs__ = 0
            module.__params__ = sum(p.numel() for p in module.parameters()
                                    if p.requires_grad)
            module.__start_time__ = 0
            module.__duration__ = 0

        self.model.apply(add_or_reset_attrs)

    def end_profile(self):
        """Ends profiling.

        The added attributes and handles are removed recursively on all the modules.
        """
        if not self.started:
            return
        self.stop_profile()
        self.started = False

        def remove_profile_attrs(module):
            if hasattr(module, "__flops__"):
                del module.__flops__
            if hasattr(module, "__macs__"):
                del module.__macs__
            if hasattr(module, "__params__"):
                del module.__params__
            if hasattr(module, "__start_time__"):
                del module.__start_time__
            if hasattr(module, "__duration__"):
                del module.__duration__

        self.model.apply(remove_profile_attrs)

    def get_total_flops(self, as_string=False):
        """Returns the total flops of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_flops = get_module_flops(self.model)
        return num_to_string(total_flops) if as_string else total_flops

    def get_total_macs(self, as_string=False):
        """Returns the total MACs of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_macs = get_module_macs(self.model)
        return macs_to_string(total_macs) if as_string else total_macs

    def get_total_duration(self, as_string=False):
        """Returns the total duration of the model forward pass.

        Args:
            as_string (bool, optional): whether to output the duration as string. Defaults to False.

        Returns:
            The latency of the model forward pass.
        """
        total_duration = get_module_duration(self.model)
        return duration_to_string(total_duration) if as_string else total_duration

    def get_total_params(self, as_string=False):
        """Returns the total parameters of the model.

        Args:
            as_string (bool, optional): whether to output the parameters as string. Defaults to False.

        Returns:
            The number of parameters in the model.
        """
        return params_to_string(
            self.model.__params__) if as_string else self.model.__params__

    def print_model_profile(self,
                            profile_step=1,
                            module_depth=-1,
                            top_modules=1,
                            detailed=True,
                            output_file=None):
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
        from os import path
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
            "Notations:\ndata parallel size (dp_size), model parallel size(mp_size),\nnumber of parameters (params), number of multiply-accumulate operations(MACs),\nnumber of floating-point operations (flops), floating-point operations per second (FLOPS),\nfwd latency (forward propagation latency), bwd latency (backward propagation latency),\nstep (weights update latency), iter latency (sum of fwd, bwd and step latency)\n"
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

        print('{:<60}  {:<8}'.format('params per gpu: ', params_to_string(total_params)))
        print('{:<60}  {:<8}'.format(
            'params of model = params per GPU * mp_size: ',
            params_to_string(total_params *
                             (self.ds_engine.mp_world_size) if self.ds_engine else 1)))

        print('{:<60}  {:<8}'.format('fwd MACs per GPU: ', macs_to_string(total_macs)))

        print('{:<60}  {:<8}'.format('fwd flops per GPU: ', num_to_string(total_flops)))

        print('{:<60}  {:<8}'.format(
            'fwd flops of model = fwd flops per GPU * mp_size: ',
            num_to_string(total_flops *
                          (self.ds_engine.mp_world_size) if self.ds_engine else 1)))

        fwd_latency = self.get_total_duration()
        if self.ds_engine and self.ds_engine.wall_clock_breakdown():
            fwd_latency = self.ds_engine.timers('forward').elapsed(False)
        print('{:<60}  {:<8}'.format('fwd latency: ', duration_to_string(fwd_latency)))
        print('{:<60}  {:<8}'.format(
            'fwd FLOPS per GPU = fwd flops per GPU / fwd latency: ',
            flops_to_string(total_flops / fwd_latency)))

        if self.ds_engine and self.ds_engine.wall_clock_breakdown():
            bwd_latency = self.ds_engine.timers('backward').elapsed(False)
            step_latency = self.ds_engine.timers('step').elapsed(False)
            print('{:<60}  {:<8}'.format('bwd latency: ',
                                         duration_to_string(bwd_latency)))
            print('{:<60}  {:<8}'.format(
                'bwd FLOPS per GPU = 2 * fwd flops per GPU / bwd latency: ',
                flops_to_string(2 * total_flops / bwd_latency)))
            print('{:<60}  {:<8}'.format(
                'fwd+bwd FLOPS per GPU = 3 * fwd flops per GPU / (fwd+bwd latency): ',
                flops_to_string(3 * total_flops / (fwd_latency + bwd_latency))))

            print('{:<60}  {:<8}'.format('step latency: ',
                                         duration_to_string(step_latency)))

            iter_latency = fwd_latency + bwd_latency + step_latency
            print('{:<60}  {:<8}'.format('iter latency: ',
                                         duration_to_string(iter_latency)))
            print('{:<60}  {:<8}'.format(
                'FLOPS per GPU = 3 * fwd flops per GPU / iter latency: ',
                flops_to_string(3 * total_flops / iter_latency)))

            samples_per_iter = self.ds_engine.train_micro_batch_size_per_gpu(
            ) * self.ds_engine.world_size
            print('{:<60}  {:<8.2f}'.format('samples/second: ',
                                            samples_per_iter / iter_latency))

        def flops_repr(module):
            params = module.__params__
            flops = get_module_flops(module)
            macs = get_module_macs(module)
            items = [
                params_to_string(params),
                "{:.2%} Params".format(params / (total_params + 1e-9)),
                macs_to_string(macs),
                "{:.2%} MACs".format(0.0 if total_macs == 0 else macs / (total_macs + 1e-9)),
            ]
            duration = get_module_duration(module)

            # TODO: make them can be optional

            items.append(duration_to_string(duration))
            items.append(
                "{:.2%} latency".format(0.0 if total_duration == 0 else duration /
                                        total_duration))
            items.append(flops_to_string(flops).lower())
            items.append(flops_to_string(0.0 if duration == 0 else flops / duration))
            items.append(module.original_extra_repr())
            return ", ".join(items)

        def add_extra_repr(module):
            flops_extra_repr = flops_repr.__get__(module)
            if module.extra_repr != flops_extra_repr:
                module.original_extra_repr = module.extra_repr
                module.extra_repr = flops_extra_repr
                assert module.extra_repr != module.original_extra_repr

        def del_extra_repr(module):
            if hasattr(module, "original_extra_repr"):
                module.extra_repr = module.original_extra_repr
                del module.original_extra_repr

        self.model.apply(add_extra_repr)

        print(
            "\n----------------------------- Aggregated Profile per GPU -----------------------------"
        )
        self.print_model_aggregated_profile(module_depth=module_depth,
                                            top_modules=top_modules)

        if detailed:
            print(
                "\n------------------------------ Detailed Profile per GPU ------------------------------"
            )
            print(
                "Each module profile is listed after its name in the following order: \nparams, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS"
            )
            print(
                "\nNote: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.\n2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.\n3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.\n"
            )
            print(self.model)

        self.model.apply(del_extra_repr)

        print(
            "------------------------------------------------------------------------------"
        )

        if output_file:
            sys.stdout = original_stdout
            f.close()

    def print_model_aggregated_profile(self, module_depth=-1, top_modules=1):
        """Prints the names of the top top_modules modules in terms of aggregated time, flops, and parameters at depth module_depth.

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
                info[curr_depth][module.__class__.__name__] = [
                    0,
                    0,
                    0,
                ]  # macs, params, time
            info[curr_depth][module.__class__.__name__][0] += get_module_macs(module)
            info[curr_depth][module.__class__.__name__][1] += module.__params__
            info[curr_depth][module.__class__.__name__][2] += get_module_duration(module)
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

            sort_macs = {
                k: macs_to_string(v[0])
                for k,
                v in sorted(info[d].items(),
                            key=lambda item: item[1][0],
                            reverse=True)[:num_items]
            }
            sort_params = {
                k: params_to_string(v[1])
                for k,
                v in sorted(info[d].items(),
                            key=lambda item: item[1][1],
                            reverse=True)[:num_items]
            }
            sort_time = {
                k: duration_to_string(v[2])
                for k,
                v in sorted(info[d].items(),
                            key=lambda item: item[1][2],
                            reverse=True)[:num_items]
            }

            print(f"depth {d}:")
            print(f"    params      - {sort_params}")
            print(f"    MACs        - {sort_macs}")
            print(f"    fwd latency - {sort_time}")


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = torch.numel(input) * out_features
    return 2 * macs, macs


def _relu_flops_compute(input, inplace=False):
    return torch.numel(input), 0


def _prelu_flops_compute(input: Tensor, weight: Tensor):
    return torch.numel(input), 0


def _elu_flops_compute(input: Tensor, alpha: float = 1.0, inplace: bool = False):
    return torch.numel(input), 0


def _leaky_relu_flops_compute(input: Tensor,
                              negative_slope: float = 0.01,
                              inplace: bool = False):
    return torch.numel(input), 0


def _relu6_flops_compute(input: Tensor, inplace: bool = False):
    return torch.numel(input), 0


def _silu_flops_compute(input: Tensor, inplace: bool = False):
    return torch.numel(input), 0


def _gelu_flops_compute(input):
    return torch.numel(input), 0


def _pool_flops_compute(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return torch.numel(input), 0


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
    has_affine = weight is not None
    if training:
        # estimation
        return torch.numel(input) * (5 if has_affine else 4), 0
    flops = torch.numel(input) * (2 if has_affine else 1)
    return flops, 0


def _layer_norm_flops_compute(
    input: Tensor,
    normalized_shape: typing.List[int],
    weight: typing.Optional[Tensor] = None,
    bias: typing.Optional[Tensor] = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return torch.numel(input) * (5 if has_affine else 4), 0


def _group_norm_flops_compute(input: Tensor,
                              num_groups: int,
                              weight: typing.Optional[Tensor] = None,
                              bias: typing.Optional[Tensor] = None,
                              eps: float = 1e-5):
    has_affine = weight is not None
    # estimation
    return torch.numel(input) * (5 if has_affine else 4), 0


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
    return torch.numel(input) * (5 if has_affine else 4), 0


def _upsample_flops_compute(input,
                            size=None,
                            scale_factor=None,
                            mode="nearest",
                            align_corners=None):
    if size is not None:
        if isinstance(size, tuple):
            return int(_prod(size)), 0
        else:
            return int(size), 0
    assert scale_factor is not None, "either size or scale_factor should be defined"
    flops = torch.numel(input)
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops * int(_prod(scale_factor))
    else:
        flops * scale_factor**len(input)
    return flops, 0


def _softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    return torch.numel(input), 0


def _embedding_flops_compute(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return 0, 0


def _dropout_flops_compute(input, p=0.5, training=True, inplace=False):
    return 0, 0


def _matmul_flops_compute(input, other, *, out=None):
    """
    Count flops for the matmul operation.
    """
    macs = _prod(input.shape) * other.shape[-1]
    return 2 * macs, macs


def _addmm_flops_compute(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(input.shape), macs


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
        print(line.lower())
        if "optimized flop" in line.lower():
            flop = int(float(line.split(":")[-1]))
            return flop, 0
    raise NotImplementedError("Unsupported einsum operation.")


def _tensor_addmm_flops_compute(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the tensor addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(self.shape), macs


def _elementwise_flops_compute(input, other, *args, **kwargs):
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


def wrapFunc(func, funcFlopCompute):
    name = func.__name__
    if inspect.isfunction(func) or inspect.isbuiltin(func):
        old_functions[(func.__module__, name)] = func
    elif inspect.ismethod(func):
        try:
            old_functions[(func.__objclass__, name)] = func
        except AttributeError:
            logger.error("{} is not correct wrapped".format(func))
        

    @functools.wraps(func)
    def newFunc(*args, **kwds):
        flops, macs = funcFlopCompute(*args, **kwds)
        if module_flop_count:
            module_flop_count[-1].append((name, flops))
        if module_mac_count and macs:
            module_mac_count[-1].append((name, macs))
        return func(*args, **kwds)

    return newFunc

def wrapWarning(func):
    name = func.__name__
    if inspect.isfunction(func) or inspect.isbuiltin(func):
        old_functions[(func.__module__, name)] = func
    elif inspect.ismethod(func):
        try:
            old_functions[(func.__objclass__, name)] = func
        except AttributeError:
            logger.error("{} is not correct wrapped".format(func))

    @functools.wraps(func)
    def newFunc(*args, **kwds):
        logger.warning("forward an unimplemented function: {}".format(name))
        return func(*args, **kwds)

    return newFunc

# for checking unimplemented part in function level
def _check_func_level_patch(pytorch_module):
    count = 0
    overridable_functions = torch.overrides.get_overridable_functions()[pytorch_module]
    for name in dir(pytorch_module):
        func = getattr(pytorch_module, name)
        if (
            func in overridable_functions and # exclude func from typing
            not (func.__module__, name) in old_functions
        ):
            count += 1
            logger.debug("[{}] Untracked function: {}".format(pytorch_module.__name__, name))
            setattr(pytorch_module, name, wrapWarning(func))
    
    logger.debug("[{}] Untracked function count: {}".format(pytorch_module.__name__, count))
 
def _patch_torch():
    # functional level
    _patch_autograd()
    _patch_fft()
    _patch_nn_functionals()
    _patch_linalg()
    _patch_special()
    
    # operator level
    _patch_tensor_methods()
    
def _patch_torchvision():
    # TODO: finish it (ops, torch, transforms), make it optional
    pass
 
def _patch_torchaudio():
    # TODO: finish it (functional), make it optional
    _check_func_level_patch(torchaudio.functional)
    pass
    
def _patch_autograd():
    # TODO: finish it
    _check_func_level_patch(torch.autograd)
    
def _patch_fft():
    # TODO: finish it
    _check_func_level_patch(torch.fft)

def _patch_nn_functionals():
    # FC
    F.linear = wrapFunc(F.linear, _linear_flops_compute)

    # convolutions
    F.conv1d = wrapFunc(F.conv1d, _conv_flops_compute)
    F.conv2d = wrapFunc(F.conv2d, _conv_flops_compute)
    F.conv3d = wrapFunc(F.conv3d, _conv_flops_compute)

    # conv transposed
    F.conv_transpose1d = wrapFunc(F.conv_transpose1d, _conv_trans_flops_compute)
    F.conv_transpose2d = wrapFunc(F.conv_transpose2d, _conv_trans_flops_compute)
    F.conv_transpose3d = wrapFunc(F.conv_transpose3d, _conv_trans_flops_compute)

    # activations
    F.relu = wrapFunc(F.relu, _relu_flops_compute)
    F.prelu = wrapFunc(F.prelu, _prelu_flops_compute)
    F.elu = wrapFunc(F.elu, _elu_flops_compute)
    F.leaky_relu = wrapFunc(F.leaky_relu, _leaky_relu_flops_compute)
    F.relu6 = wrapFunc(F.relu6, _relu6_flops_compute)
    if hasattr(F, "silu"):
        F.silu = wrapFunc(F.silu, _silu_flops_compute)
    F.gelu = wrapFunc(F.gelu, _gelu_flops_compute)

    # Normalizations
    F.batch_norm = wrapFunc(F.batch_norm, _batch_norm_flops_compute)
    F.layer_norm = wrapFunc(F.layer_norm, _layer_norm_flops_compute)
    F.instance_norm = wrapFunc(F.instance_norm, _instance_norm_flops_compute)
    F.group_norm = wrapFunc(F.group_norm, _group_norm_flops_compute)

    # poolings
    F.avg_pool1d = wrapFunc(F.avg_pool1d, _pool_flops_compute)
    F.avg_pool2d = wrapFunc(F.avg_pool2d, _pool_flops_compute)
    F.avg_pool3d = wrapFunc(F.avg_pool3d, _pool_flops_compute)
    F.max_pool1d = wrapFunc(F.max_pool1d, _pool_flops_compute)
    F.max_pool2d = wrapFunc(F.max_pool2d, _pool_flops_compute)
    F.max_pool3d = wrapFunc(F.max_pool3d, _pool_flops_compute)
    F.adaptive_avg_pool1d = wrapFunc(F.adaptive_avg_pool1d, _pool_flops_compute)
    F.adaptive_avg_pool2d = wrapFunc(F.adaptive_avg_pool2d, _pool_flops_compute)
    F.adaptive_avg_pool3d = wrapFunc(F.adaptive_avg_pool3d, _pool_flops_compute)
    F.adaptive_max_pool1d = wrapFunc(F.adaptive_max_pool1d, _pool_flops_compute)
    F.adaptive_max_pool2d = wrapFunc(F.adaptive_max_pool2d, _pool_flops_compute)
    F.adaptive_max_pool3d = wrapFunc(F.adaptive_max_pool3d, _pool_flops_compute)

    # upsample
    F.upsample = wrapFunc(F.upsample, _upsample_flops_compute)
    F.interpolate = wrapFunc(F.interpolate, _upsample_flops_compute)

    # softmax
    F.softmax = wrapFunc(F.softmax, _softmax_flops_compute)

    # embedding
    F.embedding = wrapFunc(F.embedding, _embedding_flops_compute)
    
    # not implemented
    _check_func_level_patch(F)

def _patch_linalg():
    # TODO: finish it
    _check_func_level_patch(torch.linalg)
    
def _patch_special():
    # TODO: finish it
    _check_func_level_patch(torch.special)

def _patch_tensor_methods():
    torch.matmul = wrapFunc(torch.matmul, _matmul_flops_compute)
    torch.Tensor.matmul = wrapFunc(torch.Tensor.matmul, _matmul_flops_compute)
    torch.Tensor.__matmul__ = torch.Tensor.matmul
    torch.mm = wrapFunc(torch.mm, _matmul_flops_compute)
    torch.Tensor.mm = wrapFunc(torch.Tensor.mm, _matmul_flops_compute)
    torch.bmm = wrapFunc(torch.bmm, _matmul_flops_compute)
    torch.Tensor.bmm = wrapFunc(torch.bmm, _matmul_flops_compute)

    torch.addmm = wrapFunc(torch.addmm, _addmm_flops_compute)
    torch.Tensor.addmm = wrapFunc(torch.Tensor.addmm, _tensor_addmm_flops_compute)

    torch.mul = wrapFunc(torch.mul, _elementwise_flops_compute)
    torch.Tensor.mul = wrapFunc(torch.Tensor.mul, _elementwise_flops_compute)
    torch.Tensor.__mul__ = torch.Tensor.mul
    
    torch.div = wrapFunc(torch.div, _elementwise_flops_compute)
    torch.Tensor.div = wrapFunc(torch.Tensor.div, _elementwise_flops_compute)
    torch.Tensor.__div__ = torch.Tensor.div

    torch.add = wrapFunc(torch.add, _elementwise_flops_compute)
    torch.Tensor.add = wrapFunc(torch.Tensor.add, _elementwise_flops_compute)
    torch.Tensor.__add__ = torch.Tensor.add
    
    torch.sub = wrapFunc(torch.sub, _elementwise_flops_compute)
    torch.Tensor.sub = wrapFunc(torch.Tensor.sub, _elementwise_flops_compute)
    torch.Tensor.__sub__ = torch.Tensor.sub

    torch.einsum = wrapFunc(torch.einsum, _einsum_flops_compute)


def _reload_functionals():
    for name in dir(F):
        if name in old_functions:
            setattr(F, name, old_functions[(F.__name__, name)])


def _reload_tensor_methods():
    torch.matmul = old_functions[(None, torch.matmul.__name__)]


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

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


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


MODULE_HOOK_MAPPING = {
    # RNN
    nn.RNN: _rnn_forward_hook,
    nn.GRU: _rnn_forward_hook,
    nn.LSTM: _rnn_forward_hook,
    nn.RNNCell: _rnn_cell_forward_hook,
    nn.LSTMCell: _rnn_cell_forward_hook,
    nn.GRUCell: _rnn_cell_forward_hook,
}

def _num_to_string(num, units=None, precision=2, unit="", order_names=["", "K", "M", "G", "T"], bias=0):
    order = int(min(max(np.log10(num)/3 + bias, 0), len(order_names)-1))
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


    # can not iterate over all submodules using self.model.modules()
    # since modules() returns duplicate modules only once
def get_module_flops(module):
    sum = module.__flops__
    # iterate over immediate children modules
    for child in module.children():
        sum += get_module_flops(child)
    return sum


def get_module_macs(module):
    sum = module.__macs__
    # iterate over immediate children modules
    for child in module.children():
        sum += get_module_macs(child)
    return sum


def get_module_duration(module):
    duration = module.__duration__
    if duration == 0:  # e.g. ModuleList
        for m in module.children():
            duration += m.__duration__
    return duration


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
    prof = FlopsProfiler(model)
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

        args = [input]

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
                                 output_file=output_file)

    prof.end_profile()
    if as_string:
        return number_to_string(flops), macs_to_string(macs), params_to_string(params)

    return flops, macs, params
