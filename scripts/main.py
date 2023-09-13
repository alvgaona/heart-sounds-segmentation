import os

import torch
from torch import nn

from hss.utils.training import show_progress


ROOT = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    show_progress(
        epoch=1,
        iteration=1,
        time_elapsed=1000,
        mini_batch_acc=0.5,
        mini_batch_loss=1.2,
        val_acc=0.4,
        val_loss=1.4,
        learning_rate=0.01,
    )

    show_progress(
        epoch=2,
        iteration=1200,
        time_elapsed=556,
        mini_batch_acc=0.45,
        mini_batch_loss=1.1,
        val_acc=0.4,
        val_loss=1.5,
        learning_rate=0.001,
    )
