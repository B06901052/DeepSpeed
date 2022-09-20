import os
import sys
import torch
import typing
import logging
import argparse
import importlib
import s3prl.hub as hub
import torchaudio
from time import time
from pathlib import Path

sys.path.append(os.path.realpath(os.path.join(__file__, "../../")))

from flops_profiler import get_model_profile, FlopsProfiler, number_to_string, macs_to_string, params_to_string

# logger setting
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter('[%(module)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# inputs are selected from LibriSpeech test-clean split and we choice the (82*i+1)th audio for i = 0~31 (total 2620 audios), the script used to select is "in utils/get_libri_sample.sh"
wav_paths = [
    "test-clean/672/122797/672-122797-0033.flac",# 20560
    "test-clean/4446/2275/4446-2275-0025.flac",# 34560
    "test-clean/7176/92135/7176-92135-0019.flac",# 38480
    "test-clean/2094/142345/2094-142345-0019.flac",# 42560
    "test-clean/1221/135767/1221-135767-0015.flac",# 45600
    "test-clean/61/70970/61-70970-0010.flac",# 48720
    "test-clean/8463/294825/8463-294825-0015.flac",# 52000
    "test-clean/1089/134686/1089-134686-0035.flac",# 55120
    "test-clean/1995/1836/1995-1836-0012.flac",# 59120
    "test-clean/8555/284449/8555-284449-0006.flac",# 62720
    "test-clean/121/127105/121-127105-0036.flac",# 66400
    "test-clean/237/126133/237-126133-0012.flac",# 71200
    "test-clean/7176/88083/7176-88083-0005.flac",# 75200
    "test-clean/3575/170457/3575-170457-0051.flac",# 78640
    "test-clean/7176/92135/7176-92135-0044.flac",# 82800
    "test-clean/5142/33396/5142-33396-0059.flac",# 87520
    "test-clean/260/123286/260-123286-0009.flac",# 92720
    "test-clean/2830/3980/2830-3980-0023.flac",# 98560
    "test-clean/2830/3980/2830-3980-0010.flac",# 104400
    "test-clean/4970/29095/4970-29095-0016.flac",# 111120
    "test-clean/260/123286/260-123286-0006.flac",# 118480
    "test-clean/2961/961/2961-961-0016.flac",# 125040
    "test-clean/61/70968/61-70968-0010.flac",# 132720
    "test-clean/8455/210777/8455-210777-0069.flac",# 142640
    "test-clean/908/157963/908-157963-0023.flac",# 154000
    "test-clean/4992/23283/4992-23283-0020.flac",# 166720
    "test-clean/4507/16021/4507-16021-0009.flac",# 180880
    "test-clean/1320/122617/1320-122617-0036.flac",# 197920
    "test-clean/672/122797/672-122797-0002.flac",# 217920
    "test-clean/8463/294828/8463-294828-0035.flac",# 239280
    "test-clean/2300/131720/2300-131720-0003.flac",# 268160
    "test-clean/3729/6852/3729-6852-0045.flac",# 326000
]

def synchronize(device):
    if device != "cpu":
        torch.cuda.synchronize()

# args
def get_profiling_args():
    parser=argparse.ArgumentParser()
    # upstream
    upstreams=[attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream', default="hubert", help="This is also the filename of logfile")
    parser.add_argument('--upstream_ckpt', default="", help="The ckpt path for upstream")
    # huggingface
    parser.add_argument('--from_hf_hub', action="store_true")
    parser.add_argument('--hf_org_name', type=str)
    parser.add_argument('--hf_repo_name', type=str)
    parser.add_argument('--hf_revision', type=str)
    # profiling setup
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="only for pseudo input")
    parser.add_argument('-l', '--seq_len', type=int, default=160000, help="only for pseudo input")
    parser.add_argument("--sample_rate", type=float, default=16000., help="The input sample rate")
    parser.add_argument("--pseudo_input", action="store_true", help="use torch.randn to generate pseudo input")
    parser.add_argument('-d', "--device", default="cuda")
    # information present
    parser.add_argument('-s', "--show_untracked", help="Show all untracked functions in torch and torchaudio.", action="store_true")
    parser.add_argument("--show_time", help="Show time related info.", action="store_true")
    parser.add_argument("--execution_time_only", help="Show execution time only (currently only support cuda)", action="store_true")
    parser.add_argument("-p", "--precision", type=int, default=2)
    parser.add_argument("--as_string", action="store_true", help="print result as formated string")
    parser.add_argument("--without_bucket", action="store_true", help="not divide inputs into four buckets by sequence length and report the result seperately")
    # path
    parser.add_argument("--libri_root", type=str, default="/mnt/diskb/corpora/LibriSpeech/", help="The root dir of LibriSpeech")
    parser.add_argument("--log_path", type=str, help="The path for log file storing. (not include file name)", default=os.path.join(os.path.dirname(__file__), "log/"))
    return parser.parse_args()


def s3prl_input_constructor(batch_size, seq_len, device, dtype):
    return [torch.randn(seq_len, dtype=dtype, device=device) for _ in range(batch_size)]

def pseudo_input_profiling(
    model: torch.nn.Module,
    args: argparse.Namespace,
    model_args: list=[],
    model_kwargs: dict={},
    ignore_modules: typing.List[torch.nn.Module]=[],
):
    global s3prl_input_constructor

    with torch.no_grad():
        # setup model
        model = model.eval().to(args.device)
        # get dtype of model parameters
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float
        # construct inputs
        inputs = s3prl_input_constructor(args.batch_size, args.seq_len, args.device, dtype)
        # profiling
        flops, macs, params = get_model_profile(
            model=model,
            args=[inputs, *model_args],
            kwargs=model_kwargs,
            top_modules=3,
            warm_up=10,
            as_string=args.as_string,
            output_file=os.path.join(args.log_path, "{}_pseudo.txt".format(args.upstream)),
            ignore_modules=ignore_modules,
            show_untracked=args.show_untracked,
            show_time=args.show_time,
            precision=args.precision,
        )
        del model
    
    M = macs/args.seq_len*args.sample_rate/args.batch_size if not args.as_string else "Not support --as_string"
    # summary
    logger.info("summary, l = sequence length, bs = batch size, sr = sample rate\nsum of flops: {}\nsum of macs: {}\nparams: {}\nmacs/(l/sr)/bs: {}\n".format(flops, macs, params, M))


def superb_profiling(
    model: torch.nn.Module,
    args: argparse.Namespace,
    wav_paths: str,
    model_args: list=[],
    model_kwargs: dict={},
    ignore_modules: typing.List[torch.nn.Module]=[],
):
    global pseudo_input_profiling
    # real inputs
    def load_wav(wav_path, device, dtype):
        wav, sr = torchaudio.load(os.path.join(args.libri_root, wav_path))
        assert sr == args.sample_rate, f'Sample rate mismatch: real {sr}, config {args.sample_rate}'
        wav_batch = [wav.view(-1).type(dtype).to(device)]
        return wav_batch

    with torch.no_grad():
        # initialize model
        model = model.eval().to(args.device)
        assert isinstance(model, torch.nn.Module), "model must be a PyTorch module"
        # get dtype of model parameters
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float
        # construct inputs
        samples = [load_wav(path, args.device, dtype) for path in wav_paths]
        
        # profiling start
        if not args.execution_time_only:
            prof = FlopsProfiler(model, show_untracked=args.show_untracked, show_time=args.show_time, precision=args.precision)

        # warnup
        pseudo_inputs = s3prl_input_constructor(args.batch_size, args.seq_len, args.device, dtype)
        for _ in range(10):
            _ = model(pseudo_inputs, *model_args, **model_kwargs)

        # profile seperately by bucket
        if not args.without_bucket:
            for i, bucket in enumerate(["short", "medium", "long", "longer"]):
                min_sec = samples[i<<3][0].shape[0] / args.sample_rate
                max_sec = samples[(i<<3)|7][0].shape[0] / args.sample_rate
                logger.info("bucket {}: {:.2f}~{:.2f} sec".format(bucket, min_sec, max_sec))
                
                if args.execution_time_only:
                    t = 0
                else:
                    prof.start_profile(ignore_modules)
                    synchronize(args.device)
                    pre_macs = 0
                    macs_per_seq_len = []
                    t = time()
                for inputs in samples[i<<3:(i+1)<<3]:
                    if args.execution_time_only:
                        for _ in range(10):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            _ = model(inputs, *model_args, **model_kwargs)
                            end.record()
                            synchronize(args.device)
                            t += start.elapsed_time(end)
                    else:
                        _ = model(inputs, *model_args, **model_kwargs)
                        cur_macs = prof.get_total_macs()
                        macs_per_seq_len.append((cur_macs - pre_macs) / inputs[0].shape[0])
                        pre_macs = cur_macs

                if args.execution_time_only:
                    # summary
                    logger.info(f"execution time: {t/1000/10:.4f}sec\n") # /1000 is for ms to s, /10 is for average
                else:
                    synchronize(args.device)
                    t = time() - t
                    flops = prof.get_total_flops()
                    macs = prof.get_total_macs()
                    params = prof.get_total_params()
                    prof.print_model_profile(
                        profile_step=10,
                        top_modules=3,
                        output_file=os.path.join(args.log_path, "{}_{}.txt".format(args.upstream, bucket)),
                        device=args.device,
                        input_shape=[i[0].shape for i in samples[i<<3:(i+1)<<3]]
                    )
                

                    prof.end_profile()
                
                    M, m = max(macs_per_seq_len) * args.sample_rate, min(macs_per_seq_len) * args.sample_rate
                    if args.as_string:
                        flops = number_to_string(flops, precision=args.precision)
                        macs = macs_to_string(macs, precision=args.precision)
                        params = params_to_string(params, precision=args.precision)
                        M = macs_to_string(M) + " / sec of an audio"
                        m = macs_to_string(m) + " / sec of an audio"

                    # summary
                    logger.info(f"summary, l = sequence length, bs = batch size, sr = sample rate\nsum of flops: {flops}\nsum of macs: {macs}\nparams: {params}\nrough time: {t:.3f}sec\nmacs/sec of an audio = macs/l/bs*sr\nmaximum of macs/sec of an audio: {M}\nminimum of macs/sec of an audio: {m}\n\n")
                
        # profile all
        min_sec = samples[0][0].shape[0] / args.sample_rate
        max_sec = samples[-1][0].shape[0] / args.sample_rate
        logger.info("bucket all: {:.2f}~{:.2f} sec".format(min_sec, max_sec))

        if args.execution_time_only:
            t = 0
        else:
            prof.start_profile(ignore_modules)
            synchronize(args.device)
            pre_macs = 0
            macs_per_seq_len = []
            t = time()
        for inputs in samples:
            if args.execution_time_only:
                for _ in range(10):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _ = model(inputs, *model_args, **model_kwargs)
                    end.record()
                    synchronize(args.device)
                    t += start.elapsed_time(end)
            else:
                _ = model(inputs, *model_args, **model_kwargs)
                cur_macs = prof.get_total_macs()
                macs_per_seq_len.append((cur_macs - pre_macs) / inputs[0].shape[0])
                pre_macs = cur_macs

        if args.execution_time_only:
            # summary
            logger.info(f"execution time: {t/1000/10:.4f}sec\n") # /1000 is for ms to s, /10 is for average
        else:
            synchronize(args.device)
            t = time() - t
            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            params = prof.get_total_params()
            prof.print_model_profile(
                profile_step=10,
                top_modules=3,
                output_file=os.path.join(args.log_path, "{}_all.txt".format(args.upstream)),
                device=args.device,
                input_shape=[i[0].shape for i in samples]
            )

            prof.end_profile()
                
            M, m = max(macs_per_seq_len) * args.sample_rate, min(macs_per_seq_len) * args.sample_rate
            if args.as_string:
                flops = number_to_string(flops, precision=args.precision)
                macs = macs_to_string(macs, precision=args.precision)
                params = params_to_string(params, precision=args.precision)
                M = macs_to_string(M) + " / sec of an audio"
                m = macs_to_string(m) + " / sec of an audio"

            # summary
            logger.info(f"summary, l = sequence length, bs = batch size\nsum of flops: {flops}\nsum of macs: {macs}\nparams: {params}\nrough time: {t:.3f}sec\nmacs/sec of an audio = macs/l/bs*sr\nmaximum of macs/sec of an audio: {M}\nminimum of macs/sec of an audio: {m}\n\n")
            
        


if __name__ == "__main__":
    args = get_profiling_args()
    # initialize your model here
    if args.from_hf_hub == True:
        from huggingface_hub import snapshot_download

        print(f'[Runner] - Downloading upstream model {args.upstream} from the Hugging Face Hub')
        filepath = snapshot_download(
            repo_id = f"{args.hf_org_name}/{args.hf_repo_name}",
            revision = args.hf_revision,
            use_auth_token = True
        )
        sys.path.append(filepath)

        dependencies = (Path(filepath) / 'requirements.txt').resolve()
        print("[Dependency] - The downloaded upstream model requires the following dependencies. Please make sure they are installed:")
        for idx, line in enumerate((Path(filepath) / "requirements.txt").open().readlines()):
            print(f"{idx}. {line.strip()}")
        print(f"You can install them by:\n")
        print(f"pip install -r {dependencies}\n")

        from expert import UpstreamExpert
        Upstream = UpstreamExpert
        args.upstream_ckpt = os.path.join(filepath, "model.pt")
    else:
        try:
            Upstream = getattr(hub, args.upstream)
        except AttributeError:
            print("[UpstreamExpert] - Try to import upstream locally")
            module_path = f's3prl.upstream.{args.upstream}.expert'
            Upstream = getattr(importlib.import_module(module_path), 'UpstreamExpert')
    
    if args.upstream_ckpt:
        # if initialization need ckpt
        model = Upstream(args.upstream_ckpt)
    else:
        model = Upstream()
    
    model_args = []  # forward args
    model_kwargs = {}# forward kwargs
    # profiling
    if args.pseudo_input:
        pseudo_input_profiling(model, args, model_args, model_kwargs)
    else:
        superb_profiling(model, args, wav_paths, model_args, model_kwargs)
