from typing import Union

from torch import Tensor
from helpers.klib import kdict
import pytorch_lightning as pl
import timm
from torch import optim, nn, sigmoid
from torchmetrics import Accuracy, F1


class BinaryClassifier(pl.LightningModule):
    def __init__(self, lr=0.001, num_classes=1) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model(
            'resnet50', pretrained=True, num_classes=num_classes)
        self.loss = nn.BCEWithLogitsLoss()

        shared_metrics = kdict(accuracy=Accuracy(num_classes=num_classes),
                               f1=F1(num_classes=num_classes))
        self.metrics = kdict(
            train=shared_metrics.copy(),
            val=shared_metrics.copy(),
            test=shared_metrics.copy())

    def forward(self, x):
        return self.backbone(x)

    def _log_metrics(self, step_type: str, predictions: Tensor, labels: Tensor):
        metrics = self.metrics[step_type]
        for name, metric in metrics.items():
            self.log(f"{step_type}/{name}", metric(predictions, labels))

    def _step(self, step_type: str, batch):
        data, labels = batch
        logits = self.backbone(data).squeeze(1)
        loss = self.loss(logits, labels.type_as(logits))
        self.log(f'{step_type}/loss', loss)
        self._log_metrics(step_type, sigmoid(logits), labels)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)