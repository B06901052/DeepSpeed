import torch
import fairseq
import logging
import argparse
import s3prl.hub as hub

from flops_profiler import get_model_profile


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter('[%(module)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def get_profiling_args():
    parser=argparse.ArgumentParser()
    upstreams=[attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream', default="hubert")
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-l', '--seq_len', type=int, default=160000)
    parser.add_argument('-d', "--device", default="cuda")
    return parser.parse_args()



def deepspeedProfiling(model_func, args):
    def s3prl_input_constructor(batch_size, seq_len, device, dtype):
        return [torch.randn(seq_len, dtype=dtype, device=device) for _ in range(batch_size)]


    with torch.no_grad():
        model = model_func().eval().to(args.device)
        dtype = next(model.parameters()).dtype
        inputs = s3prl_input_constructor(args.batch_size, args.seq_len, args.device, dtype)
        # TODO: add device and dtype info into the output_file
        
        flops, macs, params = get_model_profile(
            model=model,
            args=[inputs],
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=3,
            warm_up=10,
            as_string=True,
            output_file="./{}.txt".format(args.upstream),
            ignore_modules=None,
        )

if __name__ == "__main__":
    args = get_profiling_args()
    model_func=getattr(hub, args.upstream)
    deepspeedProfiling(model_func, args)
