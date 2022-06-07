# Profiling Tool for SLT2022 SUPERB Challenge

[This profiling tool](https://github.com/B06901052/DeepSpeed/tree/superb-challenge/flops_profiler) we used are mainly based on [DeepSpeed](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/profiling/flops_profiler) developed by **Microsoft**. To meet our demands, we fork it and add some additional features. For more about the profiling tool, see the original README in `flops_profilier`.

## Overview

This profiling tool profiles model by wrapping each torch function / operator with repective estimation formula, and while calling the wrapped function / oerator, the MACs and FLOPs will be accumulated (and for the case that a wrapped function in another wrapped function, drop the redundant counts). We also wrap an warning message with untracked function /operator in [here](#module-not-support-yet).

### General Estimation Rule

The profiling tool not always follows the rules below, sometimes it just use an formula with the same order, e.g. the FLOPs of most of activation functions are estimated by `torch.numel(input)`.

#### MACs (number of multiply-accumulate operations)

- <img src="https://render.githubusercontent.com/render/math?math=MACs=num(a \times b %2b c)\approx FLOPs\div 2">

#### FLOPs (number of floating-point operations)

- counted flops, each count as 1
  - arithmetic (e.g. add, sub, mul, div, pow)
  - math (e.g. sin, cos)
  - activation function (e.g. gelu, relu, lrelu)
- non-counted flops
  - assign
  - comparison
  - type converison
  - hash (e.g. embedding)
- ops not implemented yet
  - sorting-related (e.g. max, sort, topk, medium)

#### Module Not Support Yet

For the modules below, it will raise a warning while your model forwards a function in them.

- torch.special
- torch.fft
- torch.linalg
- torchaudio
- torchvision

### Supplement

- Exclude `torch.{batch, group, instance, layer}_norm`, the reasons are as below:
  1. Normally we only use the ones from `torch.nn.functional`
  2. The one from `torch.nn.functional` has chance to call the one from `torch` (C) (e.g. [here](https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#group_norm)), so we can't just compute in the one from `torch` (python) or it maybe be counted twice.
  3. rarely directly be used
- For `__rdiv__`, the computation is more costly than `__div__`, but we still count it just as `__div__`
  - Actually for `__rdiv__`, it will call `reciprocal`, `__mul__`, and `__truediv__` respectively.

### Additional Features

- Raise **warning** while forward an **untracked function/operator**
  - An untracked function/operator means that there is not a formula to directly compute its # of ops, but the # of ops could still be fully/partially counted by other tracked functions/operator in it.
- Track more functions and tensor operators
  - syntax sugar, e.g. +, -, *, /, //, **
  - basic statistics, e.g. mean, var, std
- Fix bug
  - incorrect # of ops for conv1d and conv3d
- Add `testing/unit_profiling_test.py` for unit test of module, function, or operation

## Usage

### Setup

```bash
# clone the branch that only contains profiling tool in our fork
git clone -b superb-challenge --single-branch git@github.com:B06901052/DeepSpeed.git
```

### Profiling an upstream in s3prl

```bash
cd repo-path
python testing/s3prl_profiling_test.py -u "s3prl_upstream_name" --libri_root "libri_root"
```

- The detailed result will be placed in `testing/log/{s3prl_upstream_name}.txt`

### Profiling your model

- Start from `testing/s3prl_profiling_test.py`, replace **model** by yours, add forward args and kwargs if any

  ```python
  if __name__ == "__main__":
      args = get_profiling_args()
      # initialize your model here
      model = getattr(hub, args.upstream)()
      model_args = []  # forward args
      model_kwargs = {}# forward kwargs
  ```

- Command line

  ```bash
  cd repo-path
  # add "--as_string" for making the result more readable
  python testing/s3prl_profiling_test.py -u "your_model_name" --libri_root "libri_root"
  ```

- Report the **params** and **sum of macs** to superb challenge, like the result of hubert_base as below:

  ```bash
  [s3prl_profiling_test] INFO - bucket short: 1.28~3.44 sec
  [s3prl_profiling_test] INFO - summary, l = sequence length, bs = batch size, sr = sample rate
  sum of flops: 298.81 G
  sum of macs: 149.26 GMACs
  params: 94.7 M
  rough time: 0.161sec
  macs/sec of an audio = macs/l/bs*sr
  maximum of macs/sec of an audio: 7.12 GMACs / sec of an audio
  minimum of macs/sec of an audio: 7.0 GMACs / sec of an audio


  [s3prl_profiling_test] INFO - bucket medium: 3.69~5.47 sec
  [s3prl_profiling_test] INFO - summary, l = sequence length, bs = batch size, sr = sample rate
  sum of flops: 523.02 G
  sum of macs: 261.28 GMACs
  params: 94.7 M
  rough time: 0.256sec
  macs/sec of an audio = macs/l/bs*sr
  maximum of macs/sec of an audio: 7.21 GMACs / sec of an audio
  minimum of macs/sec of an audio: 7.11 GMACs / sec of an audio


  [s3prl_profiling_test] INFO - bucket long: 5.79~8.91 sec
  [s3prl_profiling_test] INFO - summary, l = sequence length, bs = batch size, sr = sample rate
  sum of flops: 845.28 G
  sum of macs: 422.37 GMACs
  params: 94.7 M
  rough time: 0.341sec
  macs/sec of an audio = macs/l/bs*sr
  maximum of macs/sec of an audio: 7.37 GMACs / sec of an audio
  minimum of macs/sec of an audio: 7.22 GMACs / sec of an audio


  [s3prl_profiling_test] INFO - bucket longer: 9.62~20.38 sec
  [s3prl_profiling_test] INFO - summary, l = sequence length, bs = batch size, sr = sample rate
  sum of flops: 1.67 T
  sum of macs: 836.25 GMACs
  params: 94.7 M
  rough time: 0.547sec
  macs/sec of an audio = macs/l/bs*sr
  maximum of macs/sec of an audio: 7.92 GMACs / sec of an audio
  minimum of macs/sec of an audio: 7.41 GMACs / sec of an audio


  [s3prl_profiling_test] INFO - bucket all: 1.28~20.38 sec
  [s3prl_profiling_test] INFO - summary, l = sequence length, bs = batch size
  sum of flops: 3.34 T
  sum of macs: 1.67 TMACs
  params: 94.7 M
  rough time: 1.353sec
  macs/sec of an audio = macs/l/bs*sr
  maximum of macs/sec of an audio: 7.92 GMACs / sec of an audio
  minimum of macs/sec of an audio: 7.0 GMACs / sec of an audio
  ```

  - Note: This profiling tool will count all params in your model, even if some params not have not been used in forwarding. You can manually exclude those params by temporarily replace those modules with `nn.Identity()` if you want.
  - The detailed result will be placed in `testing/log/{your_model_name}_{data_bucket}.txt`
    - We sort the 32 samples by sequence length and equally divide them to 4 buckets, the names of buckets are as below:
      - short,
      - medium,
      - long,
      - longer,
      - and all for original result
