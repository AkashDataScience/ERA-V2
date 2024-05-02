#!/usr/bin/env python3
"""
Function used for visualization of data and results
Author: Shilpaj Bhalerao
Date: Jul 23, 2023
"""
# Third-Party Imports
import torch
import albumentations
from albumentations.pytorch import ToTensorV2


def _get_train_transforms():
    """Method to get train transform

    Returns:
        Object: Object to apply image augmentations
    """
    train_transforms = albumentations.Compose([
        # Add padding to image
        albumentations.PadIfNeeded(40, 40, always_apply=True), 
        # Randomly crop image
        albumentations.RandomCrop(32, 32),
        # Random horizontal flip
        albumentations.HorizontalFlip(),
        # Random cut out
        albumentations.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, 
                                     min_height=1, min_width=1,
                                     fill_value=(0.49139968, 0.48215841, 0.44653091), 
                                     mask_fill_value = None),
        # Normalize
        albumentations.Normalize([0.49139968, 0.48215841, 0.44653091], 
                                 [0.24703223, 0.24348513, 0.26158784]),
        # Convert to tensor
        ToTensorV2()])
    
    return train_transforms

def _get_test_transforms():
    """Method to get test transform

    Returns:
        Object: Object to apply image augmentations
    """
    test_transforms = albumentations.Compose(
        # Normalize
        [albumentations.Normalize([0.49139968, 0.48215841, 0.44653091], 
                                  [0.24703223, 0.24348513, 0.26158784]), 
        # Convert to tensor                          
        ToTensorV2()])
    return test_transforms


class AddGaussianNoise(object):
    """
    Class for custom augmentation strategy
    """
    def __init__(self, mean=0., std=1.):
        """
        Constructor
        """
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """
        Augmentation strategy to be implemented when called
        """
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        """
        Method to print more infor about the strategy
        """
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# Usage details
# transforms = transforms.Compose([
#     transforms.ToTensor(),
#     AddGaussianNoise(0., 1.0),
#     ])
