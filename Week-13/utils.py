#!/usr/bin/env python3
"""
Utility Script containing functions to be used for training
Author: Shilpaj Bhalerao
"""
# Standard Library Imports
import math
from typing import NoReturn

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch.optim as optim
import torch.nn.functional as F
from torch_lr_finder import LRFinder


def get_summary(model, input_size: tuple) -> NoReturn:
    """
    Function to get the summary of the model architecture
    :param model: Object of model architecture class
    :param input_size: Input data shape (Channels, Height, Width)
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    summary(network, input_size=input_size)


def get_misclassified_data(model, device, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()
    model.to(device)

    # List to store misclassified Images
    misclassified_data = []

    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:

            # Migrate the data to the device
            data, target = data.to(device), target.to(device)

            # Extract single image, label from the batch
            for image, label in zip(data, target):

                # Add batch dimension to the image
                image = image.unsqueeze(0)

                # Get the model prediction on the image
                output = model(image)

                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)

                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
    return misclassified_data


# -------------------- GradCam --------------------
def display_gradcam_output(data: list,
                           classes: list[str],
                           inv_normalize: transforms.Normalize,
                           model,
                           target_layers,
                           targets=None,
                           number_of_samples: int = 10,
                           transparency: float = 0.60):
    """
    Function to visualize GradCam output on the data
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param model: Model architecture
    :param target_layers: Layers on which GradCam should be executed
    :param targets: Classes to be focused on for GradCam
    :param number_of_samples: Number of images to print
    :param transparency: Weight of Normal image when mixed with activations
    """
    # Plot configuration
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    # Create an object for GradCam
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # Iterate over number of specified images
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        input_tensor = data[i][0]

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Get back the original image
        img = input_tensor.squeeze(0).to('cpu')
        img = inv_normalize(img)
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()

        # Mix the activations on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)

        # Display the images on the plot
        plt.imshow(visualization)
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])


def get_optimizer(model, lr, momentum=0, weight_decay=0, optimizer_type='SGD'):
    """Method to get object of stochastic gradient descent. Used to update weights.

    Args:
        model (Object): Neural Network model
        lr (float): Value of learning rate
        momentum (float): Value of momentum
        weight_decay (float): Value of weight decay
        optimizer_type (str): Type of optimizer SGD or ADAM

    Returns:
        object: Object of optimizer class to update weights
    """
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def get_StepLR_scheduler(optimizer, step_size, gamma):
    """Method to get object of scheduler class. Used to update learning rate

    Args:
        optimizer (Object): Object of optimizer
        step_size (int): Period of learning rate decay
        gamma (float): Number to multiply with learning rate

    Returns:
        object: Object of StepLR class to update learning rate
    """
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)
    return scheduler

def get_ReduceLROnPlateau_scheduler(optimizer, factor, patience):
    """Method to get object of scheduler class. Used to update learning rate

    Args:
        optimizer (Object): Object of optimizer
        factor (float): Number to multiply with learning rate
        patience (int): Number of epoch to wait

    Returns:
        object: Object of StepLR class to update learning rate
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=True)
    return scheduler

def get_OneCycleLR_scheduler(optimizer, max_lr, epochs, steps_per_epoch, max_at_epoch, anneal_strategy, div_factor, final_div_factor):
    """Method to get object of scheduler class. Used to update learning rate

    Args:
        optimizer (Object): Object of optimizer
        max_lr (float): Maximum learning rate to reach during training
        epochs (float): Total number of epoch
        steps_per_epoch (int): Total steps in an epoch
        max_at_epoch (int): Epoch to reach maximum learning rate
        anneal_strategy (string): Strategy to interpolate between minimum and maximum lr
        div_factor (int): Divisive factor to calculate intial learning rate
        final_div_factor (int): Divisive factor to calculate minimum learning rate

    Returns:
        object: Object of StepLR class to update learning rate
    """
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,  epochs=epochs,
                                              steps_per_epoch=steps_per_epoch, 
                                              pct_start=max_at_epoch/epochs,
                                              anneal_strategy=anneal_strategy,
                                              div_factor=div_factor,
                                              final_div_factor=final_div_factor)
    return scheduler

def get_criterion(loss_type='cross_entropy'):
    """Method to get loss calculation ctiterion

    Args:
        loss_type (str): Type of loss 'nll_loss' or 'cross_entropy' loss

    Returns:
        object: Object to calculate loss 
    """
    if loss_type == 'nll_loss':
        criterion = F.nll_loss
    elif loss_type == 'cross_entropy':
        criterion = F.cross_entropy
    return criterion

def get_learning_rate(model, optimizer, criterion, trainloader):
    """Method to find learning rate using LR finder.

    Args:
        model (Object): Object of model
        optimizer (Object): Object of optimizer class
        criterion (Object): Loss function
        trainloader (Object): Object of dataloader class

    Returns:
        float: Learning rate suggested by lr finder
    """
    # Create object and perform range test
    lr_finder = LRFinder(model, optimizer, criterion)
    lr_finder.range_test(trainloader, end_lr=100, num_iter=100)

    # Plot result and store suggested lr
    plot, suggested_lr = lr_finder.plot()

    # Reset model and optimizer
    lr_finder.reset()

    return suggested_lr