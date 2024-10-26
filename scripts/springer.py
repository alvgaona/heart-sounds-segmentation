import pytorch_lightning as pl
import scipy
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.segmenter import HSSegmenter
from hss.transforms import FSST


torch.set_float32_matmul_precision("high")


class HSSegmenterModule(pl.LightningModule):
    def __init__(self, input_size, batch_size, device):
        super().__init__()
        self.model = HSSegmenter(input_size=input_size, batch_size=batch_size, device=device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x.to(torch.float32))

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs.permute((0, 2, 1)), y)

        predicted = torch.max(outputs, dim=2).indices
        acc = torch.sum(predicted == y).item() / (y.size(0) * y.size(1))

        self.log_dict({"train_loss": loss, "train_acc": acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.loss_fn(outputs.permute((0, 2, 1)), y)

        predicted = torch.max(outputs, dim=2).indices
        acc = torch.sum(predicted == y).item() / (y.size(0) * y.size(1))

        self.log_dict({"val_loss": loss, "val_acc": acc})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9**epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        (
            FSST(
                1000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
                truncate_freq=(25, 200),
                stack=True,
            ),
        )
    )

    # Create full dataset
    hss_dataset = DavidSpringerHSS(
        "resources/data",
        download=True,
        framing=True,
        in_memory=True,
        transform=transform,
    )

    # Split into train, val, eval sets
    total_size = len(hss_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    eval_size = total_size - train_size - val_size

    train_dataset, val_dataset, eval_dataset = torch.utils.data.random_split(
        hss_dataset, [train_size, val_size, eval_size], generator=torch.Generator(device="cpu")
    )

    batch_size = 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device="cpu"),
        num_workers=19,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device="cpu"),
        num_workers=19,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device="cpu"),
        num_workers=19,
    )

    model = HSSegmenterModule(input_size=44, batch_size=batch_size, device=device)
    trainer = pl.Trainer(max_epochs=6, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
