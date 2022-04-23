import os
import sys
import torch
import logging
import argparse
import s3prl.hub as hub
import torchaudio

sys.path.append(os.path.realpath(os.path.join(__file__, "../../")))

from flops_profiler import get_model_profile


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter('[%(module)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


SAMPLE_RATE = 16000
wav_paths = [
    "test-clean/1320/122612/1320-122612-0008.flac",
    "test-clean/1284/1180/1284-1180-0003.flac",    
    "test-clean/1089/134686/1089-134686-0000.flac",
    "test-clean/7729/102255/7729-102255-0017.flac",
]


def get_profiling_args():
    parser=argparse.ArgumentParser()
    upstreams=[attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream', default="hubert")
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-l', '--seq_len', type=int, default=160000)
    parser.add_argument('-d', "--device", default="cuda")
    parser.add_argument('-s', "--show_untracked", help="Show all untracked functions in torch and torchaudio.", action="store_true")
    parser.add_argument("--show_time", help="Show time related info.", action="store_true")
    parser.add_argument("-p", "--precision", type=int, default=2)
    parser.add_argument("--libri_root", type=str, default="/mnt/diskb/corpora/LibriSpeech/", help="The root dir of LibriSpeech")
    parser.add_argument("--log_path", type=str, help="The path for log file storing. (not include file name)", default=os.path.join(os.path.dirname(__file__), "log/"))
    parser.add_argument("--pseudo_input", action="store_true", help="use torch.randn to generate pseudo input")
    return parser.parse_args()



def deepspeedProfiling(model_func, args):
    global wav_paths, SAMPLE_RATE
    # pseudo inputs
    def s3prl_input_constructor(batch_size, seq_len, device, dtype):
        return [torch.randn(seq_len, dtype=dtype, device=device) for _ in range(batch_size)]
    # real inputs
    def load_wav(wav_paths, device, dtype):
        wav_batch = []
        for wav_path in wav_paths:
            wav, sr = torchaudio.load(os.path.join(args.libri_root, wav_path))
            assert sr == SAMPLE_RATE, f'Sample rate mismatch: real {sr}, config {SAMPLE_RATE}'
            wav_batch.append(wav.view(-1).type(dtype).to(device))
        return wav_batch

    with torch.no_grad():
        model = model_func().eval().to(args.device)
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float
        if args.pseudo_input:
            inputs = s3prl_input_constructor(args.batch_size, args.seq_len, args.device, dtype)
        else:
            inputs = load_wav(wav_paths, args.device, dtype)
        
        flops, macs, params = get_model_profile(
            model=model,
            args=[inputs],
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=3,
            warm_up=10,
            as_string=True,
            output_file=os.path.join(args.log_path, "{}.txt".format(args.upstream)),
            ignore_modules=None,
            show_untracked=args.show_untracked,
            show_time=args.show_time,
            precision=args.precision,
        )
        logger.info("\nflops: {}\nmacs: {}\nparams: {}".format(flops, macs, params))

if __name__ == "__main__":
    args = get_profiling_args()
    model_func=getattr(hub, args.upstream)
    deepspeedProfiling(model_func, args)
