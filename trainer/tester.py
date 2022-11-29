from typing import Dict
import csv

import torch
from torch.utils.data import DataLoader
from catalyst import utils
import pandas as pd
from tqdm.auto import tqdm

from dataset.dataset import TCCDataset
from models.models import TCCModel


class Tester:
    def __init__(self, config: Dict):
        self.config = config
        self.config = config
        self.device = torch.device(self.config["device"])

        self.df = pd.read_csv(self.config["test_dataframe"])
        self.loader = self.configure_loader()

        self.model = TCCModel(self.config).to(self.config['device'])
        self.load_checkpoint()

    def configure_loader(self):
        path_prefix = self.config["test_path_prefix"]

        self.df["full_filename"] = self.df[
            "full_filename"].apply(lambda x: path_prefix + "/" + x)
        self.df["mask_fname"] = self.df["mask_fname"].apply(
            lambda x: path_prefix + "/test/" + x)

        image_paths = self.df["full_filename"]
        masks_paths = self.df["mask_fname"]

        dataset = TCCDataset(image_paths, masks_paths, mode="test",
                             device=self.device)

        loader = DataLoader(dataset, batch_size=1)

        return loader

    def load_checkpoint(self):
        checkpoint = utils.load_checkpoint(path=self.config["checkpoint"])
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=self.model,
        )

    def test(self):
        self.model.eval()
        result = []
        with torch.no_grad():
            for label, X in tqdm(self.loader):
                y_pred = torch.argmax(
                    self.model(X.to(self.config["device"]))).item()
                result.append({
                    "jpg_filename": label[0],
                    "TCC": y_pred,
                })

        with open("result.csv", "w", newline='') as result_file:
            keys = ["jpg_filename", "TCC"]
            dict_writer = csv.DictWriter(result_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(result)
