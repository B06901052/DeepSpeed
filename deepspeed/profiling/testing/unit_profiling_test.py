import os
import sys
import torch
import logging
import argparse

sys.path.append(os.path.realpath(os.path.join(__file__, "../../")))

from flops_profiler import get_model_profile
from sample_models import UnitsModel, list_all_defined_test


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter('[%(module)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('-d', "--device", default="cpu")
    parser.add_argument('-t', "--test_name", default="syntax_sugar_scalerop_test")
    parser.add_argument('-l', "--list_test", help="List all defined test.", action="store_true")
    parser.add_argument('-s', "--show_untracked", help="Show all untracked functions in torch and torchaudio.", action="store_true")
    return parser.parse_args()

"""
You can customize your test here
customized_test = {
    "test_name": "customized_test",
    "modules": [Module1, Module2],
    "input_shape": (2, 2),
    "args": {
        "arg1": ["s1", "s2"],    
        "arg2": [v1, v2],
        "kwarg1": [v1, v2],
    },
},
"""

if __name__ == "__main__":
    args = get_args()
    if args.list_test:
        list_all_defined_test()
    else:
        with torch.no_grad():
            model = UnitsModel(test_name=args.test_name)
            # model = UnitsModel(test_config=customized_test)
            model = model.eval().to(args.device)
            try:
                dtype = next(model.parameters()).dtype
            except StopIteration:
                dtype = torch.float
            
            flops, macs, params = get_model_profile(
                model=model,
                input_shape=model.test_config["input_shape"],
                print_profile=True,
                detailed=True,
                module_depth=-1,
                top_modules=3,
                warm_up=1,
                as_string=True,
                output_file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "log/{}.txt".format(args.test_name)),
                ignore_modules=None,
                show_untracked=args.show_untracked,
            )
