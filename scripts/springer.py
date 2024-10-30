import lightning.pytorch as pl
import scipy
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix
from torchvision import transforms

from hss.datasets.heart_sounds import DavidSpringerHSS
from hss.model.segmenter import HeartSoundSegmenter
from hss.transforms import FSST


class LitModel(pl.LightningModule):
    def __init__(self, input_size, batch_size, device):
        super().__init__()
        self.model = HeartSoundSegmenter(
            input_size=input_size,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size

        num_classes = 4

        self.train_acc_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_acc_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.test_acc_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_pre = Precision(task="multiclass", average="macro", num_classes=num_classes)
        self.val_pre = Precision(task="multiclass", average="macro", num_classes=num_classes)
        self.test_pre = Precision(task="multiclass", average="macro", num_classes=num_classes)

        self.train_re = Recall(task="multiclass", average="macro", num_classes=num_classes)
        self.val_re = Recall(task="multiclass", average="macro", num_classes=num_classes)
        self.test_re = Recall(task="multiclass", average="macro", num_classes=num_classes)

        self.test_confusion_mat = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x).permute((0, 2, 1))
        loss = self.loss_fn(outputs, y)

        self.train_acc(outputs, y)
        self.train_pre(outputs, y)
        self.train_re(outputs, y)
        self.train_acc_per_class(outputs, y)

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": self.train_acc,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self.log_dict(
            {
                "train_prec": self.train_pre,
                "train_re": self.train_re,
            },
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x).permute((0, 2, 1))
        loss = self.loss_fn(outputs, y)

        self.val_acc(outputs, y)
        self.val_pre(outputs, y)
        self.val_re(outputs, y)
        self.val_acc_per_class(outputs, y)

        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": self.val_acc,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self.log_dict(
            {
                "val_prec": self.val_pre,
                "val_re": self.val_re,
            },
            on_step=True,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x).permute((0, 2, 1))
        loss = self.loss_fn(outputs, y)

        self.test_acc(outputs, y)
        self.test_pre(outputs, y)
        self.test_re(outputs, y)
        self.test_acc_per_class(outputs, y)

        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": self.test_acc,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self.log_dict(
            {
                "test_prec": self.test_pre,
                "test_re": self.test_re,
            },
            on_step=True,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9**epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        (
            # Resample(35500),
            FSST(
                1000,
                window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
                truncate_freq=(25, 200),
                stack=True,
            ),
        )
    )

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
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        hss_dataset, [train_size, val_size, test_size], generator=torch.Generator(device="cpu")
    )

    batch_size = 50

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        generator=torch.Generator(device="cpu"),
        num_workers=19,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        generator=torch.Generator(device="cpu"),
        num_workers=19,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        generator=torch.Generator(device="cpu"),
        num_workers=19,
    )

    model = LitModel(input_size=44, batch_size=batch_size, device=device)
    early_stopping = EarlyStopping("val_loss", patience=6, check_finite=True)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        callbacks=[early_stopping],
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
