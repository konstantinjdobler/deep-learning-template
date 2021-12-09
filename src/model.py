from copy import deepcopy

from torch import Tensor
from torch import optim, nn, sigmoid
from torchmetrics import Accuracy, F1
import pytorch_lightning as pl
import timm
from klib import kdict


class BinaryClassifier(pl.LightningModule):
    def __init__(self, lr=0.001, num_classes=1) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model(
            'resnet50', pretrained=True, num_classes=num_classes)
        self.loss = nn.BCEWithLogitsLoss()

        shared_metrics = nn.ModuleDict(dict(accuracy=Accuracy(num_classes=num_classes),
                                            f1=F1(num_classes=num_classes)))
        self.metrics = nn.ModuleDict(dict(train_phase=deepcopy(shared_metrics),  # the `train` and `training` keywords cause an error with nn.ModuleDict
                                          dev=deepcopy(shared_metrics),
                                          test=deepcopy(shared_metrics)))

    def forward(self, x):
        return self.backbone(x)

    def _log_metrics(self, step_type: str, predictions: Tensor, labels: Tensor):
        metrics = self.metrics[step_type]
        for name, metric in metrics.items():
            metric(predictions, labels)
            self.log(f"{step_type}/{name}", metric)

    def _step(self, step_type: str, batch):
        data, labels = batch
        logits = self.backbone(data).squeeze(1)
        loss = self.loss(logits, labels.type_as(logits))
        self.log(f'{step_type}/loss', loss)
        self._log_metrics(step_type, sigmoid(logits), labels)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train_phase", batch)

    def validation_step(self, batch, batch_idx):
        return self._step("dev", batch)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
