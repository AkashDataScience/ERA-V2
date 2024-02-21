import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def is_cuda_available():
    """Method to check cuda is available or not
    """
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

def _get_train_trasforms():
    """Method to get train data transform. It perform operations on data before feeding it to
    neural network for trainig.

    Returns:
        object: Object contaning list of operations to perform on data before feeding it to neural
        network.
    """
    # Train data transformations
    train_transforms = transforms.Compose([
        # Randomly crop image from center with probability of 0.1. 
        # Used to reduce overfitting on training data
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        # Resize image for 28x28
        transforms.Resize((28, 28)),
        # Randomly rotate image between -15 to 15. Fill 0 for the area outside 
        # the rotated image. 
        # Used to reduce overfitting on training data
        transforms.RandomRotation((-15., 15.), fill=0),
        # Convert image to Tensor datatype.
        transforms.ToTensor(),
        # Normalize image. 0.1307 is mean and 0.3081 is std for our data.
        # Used to reduce bias towards a feature having hugher value then other.
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    return train_transforms

def _get_test_trasforms():
    """Method to get test data transform. It perform operations on data before feeding it to neural
    network for evaluation, testing or inference.

    Returns:
        object: Object contaning list of operations to perform on data before feeding it to neural
        network.
    """
    # Test data transformations
    test_transforms = transforms.Compose([
        # Convert image to Tensor datatype.
        transforms.ToTensor(),
        # Normalize image. 0.1307 is mean and 0.3081 is std for our data.
        # Mean and std values should be equal to train_transforms value
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    return test_transforms

def _get_data(is_train, is_download, transforms):
    """Method to get data for training or testing

    Args:
        is_train (bool): True if data is for training else false
        is_download (bool): True to download dataset from iternet
        transforms (object): Object contaning list of operations to perform on data before feeding
                             it to neural network

    Returns:
        object: Oject of dataset
    """
    data = datasets.MNIST('../data', train=is_train, download=is_download, transform=transforms)
    return data

def _get_data_loader(data, **kwargs):
    """Method to get data loader. 

    Args:
        data (object): Oject of dataset

    Returns:
        object: Object of DataLoader class used to feed data to neural network model
    """
    loader = DataLoader(data, **kwargs)
    return loader

def get_train_data_loader(batch_size):
    """Method to get data loader for trainig

    Args:
        batch_size (int): Number of images in a batch

    Returns:
        object: Object of DataLoader class used to feed data to neural network model
    """
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    train_transforms = _get_train_trasforms()
    train_data = _get_data(is_train=True, is_download=True, transforms=train_transforms)
    train_loader = _get_data_loader(data=train_data, **kwargs)
    return train_loader

def get_test_data_loader(batch_size):
    """Method to get data loader for testing

    Args:
        batch_size (int): Number of images in a batch

    Returns:
        object: Object of DataLoader class used to feed data to neural network model
    """
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    test_transforms = _get_test_trasforms()
    test_data = _get_data(is_train=False, is_download=True, transforms=test_transforms)
    test_loader = _get_data_loader(data=test_data, **kwargs)
    return test_loader

def plot_sample_images(data_loader):
    """To plot image samples from data

    Args:
        data_loader (object): Object of DataLoader class
    """
    # Store data and label from data loader
    batch_data, batch_label = next(iter(data_loader))

    fig = plt.figure()

    # Loop through data and plot images with label
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        # Plot image
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        # Add label as title
        plt.title(batch_label[i].item())
        # Remove x and y axis ticks
        plt.xticks([])
        plt.yticks([])

def GetCorrectPredCount(pPrediction, pLabels):
    """Method to get count of correct predictions

    Args:
        pPrediction (Tensor): Tensor containing prediction from model
        pLabels (Tensor): Tensor containing true labels from data

    Returns:
        int: Number of correct predictions
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def _train(model, device, train_loader, optimizer, criterion, train_losses, train_acc):
    """Method to train model for one epoch 

    Args:
        model (Object): Neural Network model
        device : torch.device indicating available device for training cuda/cpu
        train_loader (Object): Object of DataLoader class for training data
        optimizer (Object): Object of optimizer to update weights
        criterion (Object): To calculate loss
        train_losses (List): To store training loss
        train_acc (List): To store training accuracy
    """
    # Set model to training
    model.train()
    # Create progress bar
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    # Loop through batches of data
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device) # Store data and target to device
        optimizer.zero_grad() # Set gradients to zero

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item() # Update train loss

        # Backpropagation
        loss.backward() # Compute gradients
        optimizer.step() # Updates weights

        correct += GetCorrectPredCount(pred, target) # Store correct prediction count
        processed += len(data) # Store amount of data processed

        # Print results
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    # Append accuracy and losses to list
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

def _test(model, device, test_loader, criterion, test_losses, test_acc):
    """Method to test model

    Args:
        model (Object): Neural Network model
        device : torch.device indicating available device for testing cuda/cpu
        test_loader (Object): Object of DataLoader class for testing data
        criterion (Object): To calculate loss
        test_losses (List): To store testing loss
        test_acc (List): To store testing accuracy
    """
    # Set model to eval
    model.eval()

    test_loss = 0
    correct = 0

    # Disable gradient calculation
    with torch.no_grad():
        # iterate though data and calculate loss
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device) # Store data and target to device

            output = model(data) # get prediction
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target) # Store correct prediction count

    # Append accuracy and losses to list
    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    # Print results
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def get_SGD_optimizer(model, lr, momentum):
    """Method to get object of stochastic gradient descent. Used to update weights.

    Args:
        model (Object): Neural Network model
        lr (float): Value of learning rate
        momentum (float): Value of momentum

    Returns:
        object: Object of SGD class to update weights
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return optimizer

def get_scheduler(optimizer, step_size, gamma):
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

def get_criterion():
    """Method to get loss calculation ctiterion

    Returns:
        object: Object to calculate loss 
    """
    criterion = F.nll_loss
    return criterion
    
def start_training(num_epochs, model, device, train_loader, test_loader, optimizer, criterion, scheduler):
    """Method to start training the model

    Args:
        num_epochs (int): number of epochs
        model (Object): Neural Network model
        device : torch.device indicating available device for training cuda/cpu
        train_loader (Object): Object of DataLoader class for training data
        test_loader (Object): Object of DataLoader class for testing data
        optimizer (Object): Object of optimizer to update weights
        criterion (Object): To calculate loss
        scheduler (Object): To update learning rate

    Returns:
        lists: Lists containing information about losses and accuracy during training and testing
    """
    # Data to plot accuracy and loss graphs
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        # Train for one epochs
        _train(model, device, train_loader, optimizer, criterion, train_losses, train_acc)
        # Test model
        _test(model, device, test_loader, criterion, test_losses, test_acc)
        # Update learning rate
        scheduler.step()

    return train_losses, train_acc, test_losses, test_acc

def plot_accuracy_loss_graphs(train_losses, train_acc, test_losses, test_acc):
    """Method to plot loss and accuracy of training and testing

    Args:
        train_losses (List): List containing loss of model after each epoch on training data
        train_acc (List): List containing accuracy of model after each epoch on training data
        test_losses (List): List containing loss of model after each epoch on testing data
        test_acc (List): List containing accuracy of model after each epoch on testing data
    """
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    # Plot training losses
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    # Plot training accuracy
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    # Plot test losses
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    # Plot test aacuracy
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    