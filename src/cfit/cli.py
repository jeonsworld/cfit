import argparse
import re
from .core import from_hf, from_params


def cli():
    parser = argparse.ArgumentParser(description="Carbon Fit: Estimate require GPU memory")
    parser.add_argument("model_or_params", help='Model name/path for huggingface model or number of parameters')
    parser.add_argument("-p", "--precision", choices=["all", "auto", "32", "16", "8", "4"], default="all", help='Precision')
    args = parser.parse_args()

    if isinstance(args.precision, str) and args.precision not in ["all", "auto"]:
        args.precision = int(args.precision)

    model_input = args.model_or_params
    if isinstance(model_input, int) or re.match(r'^\d+(?:\.\d+)?[MBTmbt]$', model_input):
        if args.precision == "auto":
            args.precision = "all"
        print(from_params(model_input, args.precision))
    else:
        print(from_hf(model_input, args.precision))


if __name__ == '__main__':
    cli()
