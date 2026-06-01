import os
import argparse
from pathlib import Path

import utils


def main(args):
    # Reuse the unified training loop in fine_tuning.py.
    # Stage 1/2 pre-training in Uni-Sign is the same language-modeling objective,
    # differing mainly by modality (pose-only vs rgb_support) and checkpoint init.
    from fine_tuning import main as _main
    return _main(args)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign pre-training', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

