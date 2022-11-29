from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class TCCDataset(Dataset):
    def __init__(self,
                 image_paths: np.array,
                 masks_paths: np.array,
                 labels: np.array = None,
                 mode: str = 'train',
                 device: torch.device = torch.device('cpu')):
        self.image_paths = image_paths
        self.masks_paths = masks_paths
        self.labels = labels
        assert mode in ['train', 'valid',
                        'test'], "wrong mode, should be 'train', 'valid' or 'test"
        self.mode = mode
        self.device = device
        self.to_tensor_transform = transforms.ToTensor()
        self.cache = dict()

    def __len__(self) -> int:
        return self.labels.shape[0]

    def load_sample(self, file: str) -> np.array:
        if file in self.cache:
            return self.cache[file]
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.cache[file] = image
        return image

    def _prepare_sample(self, image: np.array, mask: np.array) -> np.array:
        mask_norm = np.max(mask)
        if mask_norm > 1:
            return (image * (mask / mask_norm)).astype('uint8')
        else:
            return (image * mask).astype('uint8')

    def __getitem__(self, item: int) -> Tuple[
                                            torch.Tensor, torch.Tensor] or torch.Tensor:
        img = self.load_sample(self.image_paths[item])
        mask = self.load_sample(self.masks_paths[item])
        X = self._prepare_sample(img, mask)

        if self.mode == 'train' or self.mode == 'valid':
            return self.to_tensor_transform(X).to(self.device), torch.tensor(
                self.labels[item][..., None]).to(self.device)
        else:
            img = img.split("/")[-1]
            return img, self.to_tensor_transform(X).to(self.device)
