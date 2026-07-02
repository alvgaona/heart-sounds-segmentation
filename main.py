"""Entry point — delegates to scripts/train_semi_crf.py.

Historically this ran the Semi-Markov CRF at 1000 Hz with a mismatched 50 Hz duration config
(max_duration=50 on ~2000-frame sequences), which is NaN-prone and slow (on-the-fly FSST). The
canonical, fixed trainer is scripts/train_semi_crf.py. This wrapper runs the corrected 50 Hz
10-fold cross-validation by default; any train_semi_crf.py flag can be passed through, e.g.:

    pixi run python main.py                                  # 50 Hz, 10-fold CV, GPU/CPU auto
    pixi run python main.py --folds 5 --accelerator gpu      # override anything
"""

import sys

from scripts.train_semi_crf import main, parse_args


if __name__ == "__main__":
    # With no arguments, default to the corrected 50 Hz 10-fold CV.
    if len(sys.argv) == 1:
        sys.argv += ["--downsample", "20", "--folds", "10", "--eval-fullres"]
    main(parse_args())
