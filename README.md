# Profiling Tool for SLT2022 SUPERB Challenge

[This profiling tool](https://github.com/B06901052/DeepSpeed/tree/superb-challenge/flops_profiler) we used are mainly based on [DeepSpeed](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/profiling/flops_profiler) developed by **Microsoft**. And for our demands, we fork it and add some additional features.

## Additional Features

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
python testing/s3prl_profiling_test.py -u s3prl_upstream_name
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
    python testing/s3prl_profiling_test.py -u your_model_name
    ```
 
- The detailed result will be placed in `testing/log/{your_model_name}.txt`