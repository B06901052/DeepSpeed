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

## Setup

```bash
# clone the branch that only contains profiling tool in our fork
git clone -b superb-challenge --single-branch git@github.com:B06901052/DeepSpeed.git
```

## Usage

- import in your python file

    ```python=
    import sys
    sys.path.append("repo-path")
    from flops_profiler import get_model_profile
    ```

- args and initialization, we choose four files from LibriSpeech test-clean split with different lengths.

    ```python=
    SAMPLE_RATE = 16000
    # this four files will be use to present the
    wav_paths = [
        "test-clean/1320/122612/1320-122612-0008.flac",# 126000
        "test-clean/1284/1180/1284-1180-0003.flac",    # 77360
        "test-clean/1089/134686/1089-134686-0000.flac",# 166960
        "test-clean/7729/102255/7729-102255-0017.flac",# 241760
    ]
    device = "cuda"
    dtype = torch.float
    model = your_model()
    ```

- get input

    ```python=
    def load_wav(wav_paths, device, dtype):
        wav_batch = []
        for wav_path in wav_paths:
            wav, sr = torchaudio.load(os.path.join(args.libri_root, wav_path))
            assert sr == SAMPLE_RATE, f'Sample rate mismatch: real {sr}, config {SAMPLE_RATE}'
            wav_batch.append(wav.view(-1).type(dtype).to(device))
        return wav_batch
    inputs = load_wav(wav_paths, device, dtype)
    ```

- profiling your model

    ```python=
    with torch.no_grad():
        model = model.eval().to(device)
        flops, macs, params = get_model_profile(
            model=model,
            args=[inputs],# [] is needed
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=3,
            warm_up=10,
            as_string=True,
            output_file="output_path",
            ignore_modules=None,
            show_untracked=whether_to_show_all_untracked_func,
            show_time=whether_to_show_latency_in_each_sub_module,
            precision=the_display_precision,
        )
    ```

- profile upstream in s3prl

    ```bash
    cd repo-path
    python testing/s3prl_profiling_test.py -u s3prl_upstream_name
    ```
