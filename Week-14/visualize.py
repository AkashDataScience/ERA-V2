import math
import numpy as np
import albumentations
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_inv_transforms():
    """Method to get transform to inverse the effect of normalization for ploting

    Returns:
        _Object: Object to apply image augmentations
    """
    # Normalize image
    inv_transforms = albumentations.Normalize([-0.48215841/0.24348513, -0.44653091/0.26158784, -0.49139968/0.24703223],
                                              [1/0.24348513, 1/0.26158784, 1/0.24703223], max_pixel_value=1.0)
    return inv_transforms

def plot_samples(train_loader, number_of_images):
    """Method to plot samples of augmented images

    Args:
        train_loader (Object): Object of data loader class to get images
    """
    inv_transform = get_inv_transforms()

    figure = plt.figure()
    x_count = 5
    y_count = 1 if number_of_images <= 5 else math.floor(number_of_images / x_count)
    images, labels = next(iter(train_loader))

    for index in range(1, number_of_images + 1):
        plt.subplot(y_count, x_count, index)
        plt.title(CLASS_NAMES[labels[index].numpy()])
        plt.axis('off')
        image = np.array(images[index])
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        plt.imshow(image)

def display_cifar_misclassified_data(data: list,
                                     number_of_samples: int = 10):
    """
    Function to plot images with labels
    :param data: List[Tuple(image, label)]
    :param number_of_samples: Number of images to print
    """
    fig = plt.figure(figsize=(10, 10))
    inv_transform = get_inv_transforms()

    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        img = np.array(data[i][0].squeeze().to('cpu'))
        img = np.transpose(img, (1, 2, 0))
        img = inv_transform(image=img)['image']
        plt.imshow(img)
        plt.title(r"Correct: " + CLASS_NAMES[data[i][1].item()] + '\n' + 'Output: ' + CLASS_NAMES[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])

def display_gradcam_output(data: list,
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
    inv_transform = get_inv_transforms()

    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    # Create an object for GradCam
    cam = GradCAM(model=model, target_layers=target_layers)

    # Iterate over number of specified images
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        input_tensor = data[i][0]

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Get back the original image
        img = np.array(input_tensor.squeeze(0).to('cpu'))
        img = np.transpose(img, (1, 2, 0))
        img = inv_transform(image=img)['image']
        rgb_img = np.clip(img, 0, 1)

        # Mix the activations on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)

        # Display the images on the plot
        plt.imshow(visualization)
        plt.title(r"Correct: " + CLASS_NAMES[data[i][1].item()] + '\n' + 'Output: ' + CLASS_NAMES[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])