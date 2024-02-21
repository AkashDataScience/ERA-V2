# Session 5 - Introduction to PyTorch
Convolution nueral network with fully connected layers implimentation using PyTorch.

## Overview
MNIST dataset is used for this project. It contains 60,000 images of digits from 0 to 9. We are
using convolution nueral network. The network contains 4 convolution layers and 2 fully connected
layers. ReLU is used as activation function. 

## Files
- [**model.py**](model.py)
    - This file contains architecture of convolution network. 
    - It contains **Net** Class that defines out network and the forward function. 

- [**utils.py**](utils.py)
    - This file contains other utility or helper functions for to train our model
    - **is_cuda_available**: Method to check cuda is available or not.
    - **def get_train_data_loader**: Method to get data loader for trainig.
    - **get_test_data_loader**: Method to get data loader for testing.
    - **plot_sample_images**: To plot image samples from data.
    - **GetCorrectPredCount**: Method to get count of correct predictions.
    - **get_SGD_optimizer**: Method to get object of stochastic gradient descent. Used to update
    weights.
    - **get_scheduler**: Method to get object of scheduler class. Used to update learning rate.
    - **get_criterion**: Method to get loss calculation ctiterion.
    - **start_training**: Method to start training the model.
    - **plot_accuracy_loss_graphs**: Method to plot loss and accuracy of training and testing.

- [**S5.ipynb**](S5.ipynb)
    - This is main of this project.
    - It uses function available in **model.py** and **utils.py**.
    - The flow is Import libraries -> Check cuda -> Create data loaders -> Plot samples -> Load 
    model -> Train model -> Plot accuracy and loss graphs

## Model Summary

    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
    ================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
    ================================================================
    Total params: 593,200
    Trainable params: 593,200
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.67
    Params size (MB): 2.26
    Estimated Total Size (MB): 2.94
    ----------------------------------------------------------------

## Usage
- Clone this repository.
- Run S5.ipynb