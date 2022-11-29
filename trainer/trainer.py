from catalyst import dl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from spacecutter.losses import CumulativeLinkLoss

from dataset.dataset import TCCDataset
from callbacks import AscensionCallback
from models.OrginalLogisticWithTransforms import OrdinalLogisticWithTransforms


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])

        self.df = pd.read_csv(self.config['train_dataframe'])
        self.num_classes = None
        self.train_loader, self.val_loader = self.configure_loaders()

        self.criterion = CumulativeLinkLoss(class_weights=torch.tensor(
            compute_class_weight(class_weight='balanced',
                                 classes=np.unique(self.df['observed_TCC']),
                                 y=self.df['observed_TCC'])))

        self.runner = dl.SupervisedRunner(
            input_key="features",
            output_key="logits",
            target_key="targets",
            loss_key="loss"
        )

        self.model = self.configure_model()
        self.optimizer = self.configure_optimizer()

    def configure_loaders(self):
        val_size = self.config['val_size']
        batch_size = self.config['batch_size']
        data_path_prefix = self.config['train_path_prefix']

        self.df['resized_full_filename'] = self.df[
            'resized_full_filename'].apply(
            lambda x: data_path_prefix + '/' + x)
        self.df['resized_mask_fname'] = self.df['resized_mask_fname'].apply(
            lambda x: data_path_prefix + '/' + x)

        self.num_classes = self.df['observed_TCC'].unique().shape[0]
        data_classes = []
        for class_i in range(self.num_classes):
            data_classes.append(self.df[self.df['observed_TCC'] == class_i])

        num_valid_per_class = int(
            self.df.shape[0] * val_size // self.num_classes)
        class_lengths = [len(s) for s in data_classes]
        class_random_indexes = [
            np.random.choice(length, size=length, replace=False) for length in
            class_lengths]

        valid_data = [class_data.iloc[indexes[:num_valid_per_class]] for
                      class_data, indexes in
                      zip(data_classes, class_random_indexes)]
        train_data = [class_data.iloc[indexes[num_valid_per_class:]] for
                      class_data, indexes in
                      zip(data_classes, class_random_indexes)]

        valid_data = pd.concat(valid_data, axis=0).reset_index()
        train_data = pd.concat(train_data, axis=0).reset_index()

        image_paths_train = train_data['resized_full_filename']
        masks_paths_train = train_data['resized_mask_fname']
        y_train = train_data['observed_TCC']
        image_paths_valid = valid_data['resized_full_filename']
        masks_paths_valid = valid_data['resized_mask_fname']

        y_valid = valid_data['observed_TCC']
        val_dataset = TCCDataset(image_paths_valid, masks_paths_valid,
                                 y_valid, mode='valid',
                                 device=self.device)
        train_dataset = TCCDataset(image_paths_train, masks_paths_train,
                                   y_train, mode='train',
                                   device=self.device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False)

        return train_loader, val_loader

    def configure_model(self):
        predictor = torchvision.models.resnet152(pretrained=True)
        # num_features -- это размерность вектора фич, поступающего на вход FC-слою
        num_features = 2048
        num_classes = 9
        # Заменяем Fully-Connected слой на наш линейный классификатор
        predictor.fc = nn.Linear(num_features, 1)

        return OrdinalLogisticWithTransforms(predictor, num_classes).to(
            self.device)

    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config[
            'learning_rate'])

    def train(self):
        loaders = {'train': self.train_loader, 'valid': self.val_loader}

        self.runner.train(
            engine=dl.DeviceEngine(self.config['device']),
            loaders=loaders,
            model=self.model, criterion=self.criterion,
            optimizer=self.optimizer,
            num_epochs=1000,
            callbacks=[
                dl.AccuracyCallback(input_key="logits", target_key="targets"),
                AscensionCallback(),
                dl.OptimizerCallback(metric_key='loss', accumulation_steps=4),
                dl.CheckpointCallback(logdir="./checkpoints",
                                      loader_key="valid", metric_key="loss",
                                      minimize=True),
                dl.EarlyStoppingCallback(patience=2, loader_key="valid",
                                         metric_key="loss", minimize=True),
            ],
            loggers={
                "wandb": dl.WandbLogger(project="TCC-Estimation")
            },
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
            load_best_on_end=True,
        )
