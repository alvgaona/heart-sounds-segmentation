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
    # Apply the corrected 50 Hz 10-fold CV defaults for any flag the user didn't set explicitly,
    # so e.g. `main.py --accelerator gpu` still runs the CV rather than falling back to a 1000 Hz split.
    def _default(flag: str, *values: str) -> None:
        if flag not in sys.argv:
            sys.argv.extend([flag, *values])

    _default("--downsample", "20")
    _default("--folds", "10")
    _default("--eval-fullres")
    main(parse_args())
