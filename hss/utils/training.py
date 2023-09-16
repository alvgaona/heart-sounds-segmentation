from datetime import timedelta
from typing import Optional

import pandas as pd


def show_progress(
    *,
    epoch: int,
    iteration: int,
    time_elapsed: float,
    mini_batch_size: int,
    mini_batch_acc: float,
    mini_batch_loss: float,
    learning_rate: float,
    val_acc: Optional[float] = None,
    val_loss: Optional[float] = None,
) -> None:
    """
    Shows the progress in a well-organized format.

    Args:
        epoch (int): the current epoch.
        iteration (int): the current iteration.
        time_elapsed (float): time elapsed passed in seconds.
        mini_batch_size (int): size of the mini-batch
        mini_batch_acc (float): the mini-batch accuracy.
        mini_batch_loss (float): the mini-batch loss value.
        learning_rate (float): the base learning rate used in the iteration.
        val_acc (float, optional): the validation accuracy.
        val_loss (float, optional): the validation loss value.
    """
    if iteration < 0:
        raise ValueError(
            f"Iteration {iteration} should not be negative. Please pass a non-negative number."
        )

    td = timedelta(seconds=time_elapsed)
    hms = str(td).split(":")
    table = {
        "Epoch": str(epoch),
        "Iteration": str(iteration),
        "Time Elapsed (hh:mm:ss)": hms[0] + ":" + hms[1] + ":" + f"{float(hms[2]):.2f}",
        "Mini-batch Accuracy": f"{(mini_batch_acc * 100):.4f}" + "%",
        "Mini-batch Loss": f"{mini_batch_loss:.4f}",
    }

    if val_acc and val_loss:
        table["Validation Accuracy"] = f"{(val_acc * 100):.4f}" + "%"
        table["Validation Loss"] = f"{val_loss:.4f}"

    table["Learning Rate"] = f"{learning_rate:.4f}"

    print("\r", end="")

    header_str = "|"
    for header in table.keys():
        header_str += f" {header : ^5} |"
    if iteration == mini_batch_size:
        print("|" + "=" * (len(header_str) - 2) + "|")
        print(header_str)

    values_str = "|"
    for header, value in table.items():
        values_str += f" {value : >{len(header)}} |"
    print(values_str)
    print(f"|{'=' * (len(header_str) - 2)}|", end="")
