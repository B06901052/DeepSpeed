import os
import sys
import torch
import fairseq
import logging
import argparse
import s3prl.hub as hub

sys.path.append(os.path.realpath(os.path.join(__file__, "../../")))

from flops_profiler import get_model_profile
from sample_models import UnitsModel


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter('[%(module)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_profiling_args():
    parser=argparse.ArgumentParser()
    upstreams=[attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream', default="toy")
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-l', '--seq_len', type=int, default=160000)
    parser.add_argument('-d', "--device", default="cuda")
    return parser.parse_args()



def deepspeedProfiling(model_func, args):
    with torch.no_grad():
        model = model_func().eval().to(args.device)
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float
        # TODO: add device and dtype info into the output_file
        
        input_shape = (2,2,2,2)
        flops, macs, params = get_model_profile(
            model=model,
            input_shape=input_shape,
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=3,
            warm_up=1,
            as_string=True,
            output_file="./{}.txt".format(args.upstream),
            ignore_modules=None,
        )

if __name__ == "__main__":
    args = get_profiling_args()
    model_func = ToyModel
    deepspeedProfiling(model_func, args)
