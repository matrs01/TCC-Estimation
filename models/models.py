from typing import Dict

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from catalyst import utils
from spacecutter.models import OrdinalLogisticModel
from efficientnet_pytorch import EfficientNet
import imgaug as ia
import imgaug.augmenters as iaa


class TCCModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.num_classes = 9
        self.config = config

        self.train_transforms = iaa.RandAugment(n=(0, 10), m=(5, 25))
        self.valid_transforms = iaa.Identity()
        self.norm = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        self.model = self.configure_model()

    def configure_model(self):
        predictor = None
        if self.config["base_model"].startswith("resnet"):
            predictor = torchvision.models.resnet152(pretrained=True)
            predictor.fc = nn.Linear(in_features=2048,
                                     out_features=1)
        elif self.config["base_model"].startswith("efficientnet"):
            predictor = EfficientNet.from_pretrained(self.config["base_model"])
            predictor._fc = nn.Linear(in_features=2560,
                                      out_features=1)

        return OrdinalLogisticModel(predictor, self.num_classes)

    def freeze_extractor(self):
        utils.set_requires_grad(self.model, False)

        if self.config["base_model"].startswith("resnet"):
            utils.set_requires_grad(self.model.predictor.fc, True)
        elif self.config["base_model"].startswith("efficientnet"):
            utils.set_requires_grad(self.model.predictor._fc, True)

        utils.set_requires_grad(self.model.link, True)

    def unfreeze_extractor(self):
        utils.set_requires_grad(self.model, True)

    def forward(self, x: torch.Tensor):
        if self.training:
            x = self.train_transforms(images=x.cpu().detach().numpy())
        else:
            x = self.valid_transforms(images=x.cpu().detach().numpy())

        x = np.transpose(x, (0, 3, 1, 2))
        x = self.norm(torch.from_numpy(x / 255).float().to(
            self.config["device"]))

        return self.model(x)
