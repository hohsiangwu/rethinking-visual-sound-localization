import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from ..modules.resnet import BasicBlock
from ..modules.resnet import resnet18
from ..modules.resnet import ResNetSpec


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPLoss1D(nn.Module):
    def __init__(self):
        super(CLIPLoss1D, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_image = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        batch_size = image_features.shape[0]
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
        return (
            self.loss_image(logits_per_image, ground_truth)
            + self.loss_text(logits_per_text, ground_truth)
        ) / 2


class LightningBase(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        return {
            "avg_val_loss": avg_loss,
            "log": {"val_loss": avg_loss},
            "progress_bar": {"val_loss": avg_loss},
        }

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss", avg_loss, prog_bar=True)
        return {
            "avg_test_loss": avg_loss,
            "log": {"test_loss": avg_loss},
            "progress_bar": {"test_loss": avg_loss},
        }

    def configure_optimizers(self):
        if self.args["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.args.learning_rate, momentum=0.9
            )
        elif self.args["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.args["learning_rate"]
            )
        else:
            assert False
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.args["lr_scheduler_patience"],
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class RCGrad(LightningBase):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.args = args
        self.image_encoder = resnet18(modal="vision", pretrained=True)
        self.audio_encoder = ResNetSpec(
            BasicBlock,
            [2, 2, 2, 2],
            pool="avgpool",
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
        )
        self.loss_fn = CLIPLoss1D()

    def forward(self, audio, image):
        audio_output = self.audio_encoder(audio.float())
        image_output = self.image_encoder(image.float())
        return audio_output, image_output

    def step(self, batch, batch_idx):
        audio, images = batch
        audio_out, image_out = self.forward(audio, images)
        loss = self.loss_fn(audio_out, image_out)
        return loss
