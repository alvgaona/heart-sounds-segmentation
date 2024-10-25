from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from rich.console import Console
from rich.table import Table


@dataclass
class TrainingMetrics:
    epoch: int
    iteration: int
    time_elapsed: float
    mini_batch_size: int
    mini_batch_acc: float
    mini_batch_loss: float
    learning_rate: float
    val_acc: Optional[float] = None
    val_loss: Optional[float] = None


class ProgressTracker:
    def __init__(self):
        self.console = Console()
        self.header_shown = False

    def show_progress(
        self,
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
        Shows the progress in a well-organized format using Rich library for better formatting.

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
            raise ValueError(f"Iteration {iteration} should not be negative. Please pass a non-negative number.")

        # Create metrics object for better organization
        metrics = TrainingMetrics(
            epoch=epoch,
            iteration=iteration,
            time_elapsed=time_elapsed,
            mini_batch_size=mini_batch_size,
            mini_batch_acc=mini_batch_acc,
            mini_batch_loss=mini_batch_loss,
            learning_rate=learning_rate,
            val_acc=val_acc,
            val_loss=val_loss,
        )

        table = Table(show_header=not self.header_shown, header_style="bold magenta")

        # Format time
        td = timedelta(seconds=metrics.time_elapsed)
        time_str = str(td).split(":")
        formatted_time = f"{time_str[0]}:{time_str[1]}:{float(time_str[2]):.2f}"

        # Add columns
        table.add_column("Epoch")
        table.add_column("Iteration")
        table.add_column("Time (hh:mm:ss)")
        table.add_column("Mini-batch Acc")
        table.add_column("Mini-batch Loss")
        if metrics.val_acc is not None and metrics.val_loss is not None:
            table.add_column("Val Acc")
            table.add_column("Val Loss")
        table.add_column("Learning Rate")

        # Add row
        row = [
            str(metrics.epoch),
            str(metrics.iteration),
            formatted_time,
            f"{(metrics.mini_batch_acc * 100):.4f}%",
            f"{metrics.mini_batch_loss:.4f}",
        ]

        if metrics.val_acc is not None and metrics.val_loss is not None:
            row.extend(
                [
                    f"{(metrics.val_acc * 100):.4f}%",
                    f"{metrics.val_loss:.4f}",
                ]
            )

        row.append(f"{metrics.learning_rate:.4f}")

        table.add_row(*row)

        # Only show newline before first header
        if not self.header_shown:
            self.console.print("\n")
            self.header_shown = True

        self.console.print(table)
