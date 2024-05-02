import os
import numpy as np
import albumentations
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR10
from pytorch_lightning import LightningDataModule
from augmentations import _get_train_transforms, _get_test_transforms

class AlbumDataset(Dataset):
    def __init__(self, dataset, trasforms=None):
        """Constructor for CIFAR10Data

        Args:
            dataset (Object): CIFAR10 Data object
            transforms (Object, optional): Object to augment data
        """
        self.dataset = dataset
        self.transforms = trasforms
        self.classes = dataset.classes

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        """Method to get image and lable of CIFAR10

        Args:
            index (int): Index of item required

        Returns:
            Tensor: Image and label in tensor format
        """
        image, label = self.dataset[index]
        
        image = np.array(image)

        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, label
    
class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None) -> None:
        self.train_transforms = albumentations.Compose(_get_train_transforms())
        self.test_transforms = albumentations.Compose(_get_test_transforms())

        # Assign train/val datasets 
        if stage == "fit" or stage is None:
            cifar_data = AlbumDataset(CIFAR10(self.data_dir, train=True),  self.train_transforms)
            self.cifar_train, self.cifar_val = random_split(cifar_data, [45000, 5000])

        # Assign test dataset
        if stage == "test" or stage is None:
            self.cifar_test = AlbumDataset(CIFAR10(self.data_dir, train=False), self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers)