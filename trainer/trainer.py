from catalyst import dl
from catalyst import utils
from catalyst.data.sampler import BalanceClassSampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from spacecutter.losses import CumulativeLinkLoss

from dataset.dataset import TCCDataset
from trainer.callbacks import AscensionCallback
from trainer.loggers import WandbLoggerWithPauses
from models.models import TCCModel
from runners.runners import FineTuningRunner


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config["device"])

        self.df = pd.read_csv(self.config["train_dataframe"])
        self.num_classes = None
        self.train_loader, self.val_loader = self.configure_loaders()

        self.criterion = CumulativeLinkLoss(class_weights=torch.tensor(
            compute_class_weight(class_weight="balanced",
                                 classes=np.unique(self.df["observed_TCC"]),
                                 y=self.df["observed_TCC"])))

        # TODO передавать модель в конструктор
        self.runner = FineTuningRunner(
            config=self.config,
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss"
        )

        self.model = TCCModel(self.config).to(self.device)
        self.optimizer = self.configure_optimizer()

        if self.config["from_checkpoint"]:
            self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint = utils.load_checkpoint(path=self.config["from_checkpoint"])
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
        )

    def configure_loaders(self):
        batch_size = self.config["batch_size"]
        data_path_prefix = self.config["train_path_prefix"]

        self.df["resized_full_filename"] = self.df[
            "resized_full_filename"].apply(
            lambda x: data_path_prefix + "/" + x)
        self.df["resized_mask_fname"] = self.df["resized_mask_fname"].apply(
            lambda x: data_path_prefix + "/" + x)

        filenames = self.df["resized_full_filename"].to_numpy()
        masks = self.df["resized_mask_fname"].to_numpy()
        labels = self.df["observed_TCC"].to_numpy()

        val_size = int(len(filenames) * self.config["val_size"])

        val_indices = np.random.choice(len(filenames), val_size, False)
        train_indices = np.setdiff1d(np.arange(len(filenames)), val_indices,
                                     True)

        train_dataset = TCCDataset(
            filenames[train_indices], masks[train_indices],
            labels[train_indices], mode="train"
        )
        val_dataset = TCCDataset(
            filenames[val_indices], masks[val_indices],
            labels[val_indices], mode="valid"
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                sampler=BalanceClassSampler(
                                    labels[val_indices], "upsampling",
                                ))

        return train_loader, val_loader

    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.config["learning_rate"],
                                weight_decay=self.config["weight_decay"])

    def train(self):
        loaders = {"train": self.train_loader, "valid": self.val_loader}

        self.runner.train(
            loaders=loaders,
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            num_epochs=4,
            callbacks=[
                dl.OptimizerCallback(metric_key="loss",
                                     accumulation_steps=3,
                                     grad_clip_fn=nn.utils.clip_grad_norm_,
                                     grad_clip_params={"max_norm": 20}),
                AscensionCallback(),
                dl.AccuracyCallback(input_key="logits", target_key="targets"),
                dl.CheckpointCallback(logdir="./checkpoints",
                                      loader_key="valid", metric_key="loss",
                                      minimize=True),
            ],
            loggers={
                "wandb": WandbLoggerWithPauses(
                    self.model, self.config, project="TCC-Estimation")
            },
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
            load_best_on_end=True,
        )
