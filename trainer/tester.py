from typing import Dict
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from catalyst import utils
import pandas as pd

from dataset.dataset import TCCDataset
from models.OrginalLogisticWithTransforms import OrdinalLogisticWithTransforms


class Tester:
    def __init__(self, config: Dict):
        self.config = config
        self.config = config
        self.device = torch.device(self.config["device"])

        self.df = pd.read_csv(self.config["dataframe"])
        self.loader = self.configure_loader()

        self.model = self.configure_model()

    def configure_loader(self):
        path_prefix = self.config["test_path_prefix"]

        self.df["full_filename"] = self.df[
            "resized_full_filename"].apply(lambda x: path_prefix + "/" + x)
        self.df["resized_mask_fname"] = self.df["resized_mask_fname"].apply(
            lambda x: path_prefix + "/" + x)

        image_paths = self.df["resized_full_filename"]
        masks_paths = self.df["resized_mask_fname"]

        dataset = TCCDataset(image_paths, masks_paths, mode="test",
                             device=self.device)

        loader = DataLoader(dataset, batch_size=1)

        return loader

    def configure_model(self):
        predictor = torchvision.models.resnet152(pretrained=True)
        # num_features -- это размерность вектора фич, поступающего на вход FC-слою
        num_features = 2048
        num_classes = 9
        # Заменяем Fully-Connected слой на наш линейный классификатор
        predictor.fc = nn.Linear(num_features, 1)

        model = OrdinalLogisticWithTransforms(predictor, num_classes).to(
            self.device)

        checkpoint = utils.load_checkpoint(path=self.config["checkpoint"])
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=model,
        )

        return model

    def test(self):
        self.model.eval()
        result = []
        with torch.no_grad:
            for label, X in self.loader:
                y_pred = self.model(self.loader).item()
                result.append({
                    "jpg_filename": label,
                    "TCC": y_pred,
                })

        with open("result.csv", "w", newline='') as result_file:
            keys = ["jpg_filename", "TCC"]
            dict_writer = csv.DictWriter(result_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(result)
