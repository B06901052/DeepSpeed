import os
import sys
import torch
import typing
import logging
import argparse
import s3prl.hub as hub
import torchaudio

sys.path.append(os.path.realpath(os.path.join(__file__, "../../")))

from flops_profiler import get_model_profile, FlopsProfiler, number_to_string, macs_to_string, params_to_string

# logger setting
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter('[%(module)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# config
SAMPLE_RATE = 16000
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

# args
def get_profiling_args():
    parser=argparse.ArgumentParser()
    upstreams=[attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream', default="hubert", help="This is also the filename of logfile")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="only for pseudo input")
    parser.add_argument('-l', '--seq_len', type=int, default=160000, help="only for pseudo input")
    parser.add_argument('-d', "--device", default="cuda")
    parser.add_argument('-s', "--show_untracked", help="Show all untracked functions in torch and torchaudio.", action="store_true")
    parser.add_argument("--show_time", help="Show time related info.", action="store_true")
    parser.add_argument("-p", "--precision", type=int, default=2)
    parser.add_argument("--libri_root", type=str, default="/mnt/diskb/corpora/LibriSpeech/", help="The root dir of LibriSpeech")
    parser.add_argument("--log_path", type=str, help="The path for log file storing. (not include file name)", default=os.path.join(os.path.dirname(__file__), "log/"))
    parser.add_argument("--pseudo_input", action="store_true", help="use torch.randn to generate pseudo input")
    parser.add_argument("--as_string", action="store_true", help="print result as formated string")
    return parser.parse_args()


def s3prl_input_constructor(batch_size, seq_len, device, dtype):
    return [torch.randn(seq_len, dtype=dtype, device=device) for _ in range(batch_size)]

def pseudo_input_profiling(
    model: torch.nn.Module,
    args: argparse.Namespace
):
    global SAMPLE_RATE, s3prl_input_constructor

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
            args=[inputs],
            top_modules=3,
            warm_up=10,
            as_string=args.as_string,
            output_file=os.path.join(args.log_path, "{}.txt".format(args.upstream)),
            show_untracked=args.show_untracked,
            show_time=args.show_time,
            precision=args.precision,
        )
        del model
        return flops, macs, params


def superb_profiling(
    model: torch.nn.Module,
    args: argparse.Namespace,
    wav_paths: str,
    ignore_modules: typing.List[torch.nn.Module]=[]
):
    global SAMPLE_RATE, pseudo_input_profiling
    # real inputs
    def load_wav(wav_path, device, dtype):
        wav, sr = torchaudio.load(os.path.join(args.libri_root, wav_path))
        assert sr == SAMPLE_RATE, f'Sample rate mismatch: real {sr}, config {SAMPLE_RATE}'
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
        prof = FlopsProfiler(model, show_untracked=args.show_untracked, show_time=args.show_time, precision=args.precision)

        # warnup
        pseudo_inputs = s3prl_input_constructor(args.batch_size, args.seq_len, args.device, dtype)
        for _ in range(10):
            _ = model(pseudo_inputs)

        prof.start_profile(ignore_modules)

        pre_macs = 0
        macs_per_seq_len = []
        for inputs in samples:
            _ = model(inputs)
            cur_macs = prof.get_total_macs()
            macs_per_seq_len.append((cur_macs - pre_macs) / inputs[0].shape[0])
            pre_macs = cur_macs

        flops = prof.get_total_flops()
        macs = prof.get_total_macs()
        params = prof.get_total_params()
        prof.print_model_profile(
            profile_step=10,
            top_modules=3,
            output_file=os.path.join(args.log_path, "{}.txt".format(args.upstream)),
            device=args.device,
            input_shape=[i[0].shape for i in samples]
        )

        prof.end_profile()
        del model
        if args.as_string:
            return number_to_string(flops, precision=args.precision), macs_to_string(macs, precision=args.precision), params_to_string(params, precision=args.precision), max(macs_per_seq_len), min(macs_per_seq_len)

        return flops, macs, params, max(macs_per_seq_len), min(macs_per_seq_len)


if __name__ == "__main__":
    args = get_profiling_args()
    # initialize your model here
    model = getattr(hub, args.upstream)()
    # profiling
    if args.pseudo_input:
        flops, macs, params = pseudo_input_profiling(model, args)
        M, m = None, None
    else:
        flops, macs, params, M, m = superb_profiling(model, args, wav_paths)
    # summary
    logger.info("summary, l = sequence length, bs = batch size\nsum of flops: {}\nsum of macs: {}\nparams: {}\nmaximum of macs/l/bs: {}\nminimum of macs/l/bs: {}\n\n".format(flops, macs, params, M, m))
