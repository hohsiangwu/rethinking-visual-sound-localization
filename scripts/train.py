import os

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from rethinking_visual_sound_localization.training.data import AudioVisualDataset
from rethinking_visual_sound_localization.training.data import worker_init_fn
from rethinking_visual_sound_localization.training.model import RCGrad

if __name__ == "__main__":
    args = {
        "num_gpus": 1,
        "batch_size": 256,
        "learning_rate": 0.001,
        "lr_scheduler_patience": 5,
        "early_stopping_patience": 10,
        "optimizer": "Adam",
        "num_workers": 8,
        "random_state": 2021,
        "args.debug": False,
    }
    seed_everything(args["random_state"])

    project_root = "PATH_TO_PROJECT_ROOT"
    os.makedirs(project_root, exist_ok=True)
    tensorboard_logger = TensorBoardLogger(save_dir="{}/logs/".format(project_root))

    dirpath = "{}/models/".format(project_root)
    filename = "{epoch}-{val_loss:.4f}"

    trainer = Trainer(
        logger=tensorboard_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args["early_stopping_patience"]),
            ModelCheckpoint(
                dirpath=dirpath, filename=filename, monitor="val_loss", save_top_k=-1
            ),
        ],
        gpus=args["num_gpus"],
        accelerator="dp",
        max_epochs=100,
    )
    train_loader = DataLoader(
        AudioVisualDataset(data_root="PATH_TO_DATA_ROOT", split="train", duration=5),
        num_workers=args["num_workers"],
        batch_size=args["batch_size"],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        AudioVisualDataset(data_root="PATH_TO_DATA_ROOT", split="valid", duration=5),
        num_workers=args["num_workers"],
        batch_size=args["batch_size"],
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    rc_grad = RCGrad(args)
    trainer.fit(rc_grad, train_loader, valid_loader)
