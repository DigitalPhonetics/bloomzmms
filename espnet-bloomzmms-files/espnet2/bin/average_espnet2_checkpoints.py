#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import torch

from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.train.reporter import Reporter


def main():
    output_dir = Path(args.output_dir)
    reporter = Reporter()
    states = torch.load(
        output_dir / "checkpoint.pth",
        map_location="cpu",
    )
    reporter.load_state_dict(states["reporter"])
    average_nbest_models(
        reporter=reporter,
        output_dir=output_dir,
        best_model_criterion=[args.criterion.split()],
        nbest=args.num,
    )


def get_parser():
    parser = argparse.ArgumentParser(
        description="average models from snapshots in training output directory"
    )
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--criterion", default="valid acc max", type=str)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
