# Assignment
1. Train any (pytorch or lightning) model on SageMaker and share the screenshot of the final logs.
2. Train any (pytorch or lightning) model on EC2 and share the screenshot of the final logs.

# Introduction
The goal of this assignment is to use Sagemaker and EC2 to train ResNet Model.

## SageMaker training
![SageMaker](./images/Sagemaker_training.png)

## Model summary

        BatchNorm2d-2           [-1, 64, 32, 32]             128
             Conv2d-3           [-1, 64, 32, 32]          36,864
        BatchNorm2d-4           [-1, 64, 32, 32]             128
             Conv2d-5           [-1, 64, 32, 32]          36,864
        BatchNorm2d-6           [-1, 64, 32, 32]             128
         BasicBlock-7           [-1, 64, 32, 32]               0
             Conv2d-8           [-1, 64, 32, 32]          36,864
        BatchNorm2d-9           [-1, 64, 32, 32]             128
            Conv2d-10           [-1, 64, 32, 32]          36,864
       BatchNorm2d-11           [-1, 64, 32, 32]             128
        BasicBlock-12           [-1, 64, 32, 32]               0
            Conv2d-13          [-1, 128, 16, 16]          73,728
       BatchNorm2d-14          [-1, 128, 16, 16]             256
            Conv2d-15          [-1, 128, 16, 16]         147,456
       BatchNorm2d-16          [-1, 128, 16, 16]             256
            Conv2d-17          [-1, 128, 16, 16]           8,192
       BatchNorm2d-18          [-1, 128, 16, 16]             256
        BasicBlock-19          [-1, 128, 16, 16]               0
            Conv2d-20          [-1, 128, 16, 16]         147,456
       BatchNorm2d-21          [-1, 128, 16, 16]             256
            Conv2d-22          [-1, 128, 16, 16]         147,456
       BatchNorm2d-23          [-1, 128, 16, 16]             256
        BasicBlock-24          [-1, 128, 16, 16]               0
            Conv2d-25            [-1, 256, 8, 8]         294,912
       BatchNorm2d-26            [-1, 256, 8, 8]             512
            Conv2d-27            [-1, 256, 8, 8]         589,824
       BatchNorm2d-28            [-1, 256, 8, 8]             512
            Conv2d-29            [-1, 256, 8, 8]          32,768
       BatchNorm2d-30            [-1, 256, 8, 8]             512
        BasicBlock-31            [-1, 256, 8, 8]               0
            Conv2d-32            [-1, 256, 8, 8]         589,824
       BatchNorm2d-33            [-1, 256, 8, 8]             512
            Conv2d-34            [-1, 256, 8, 8]         589,824
       BatchNorm2d-35            [-1, 256, 8, 8]             512
        BasicBlock-36            [-1, 256, 8, 8]               0
            Conv2d-37            [-1, 512, 4, 4]       1,179,648
       BatchNorm2d-38            [-1, 512, 4, 4]           1,024
            Conv2d-39            [-1, 512, 4, 4]       2,359,296
       BatchNorm2d-40            [-1, 512, 4, 4]           1,024
            Conv2d-41            [-1, 512, 4, 4]         131,072
       BatchNorm2d-42            [-1, 512, 4, 4]           1,024
        BasicBlock-43            [-1, 512, 4, 4]               0
            Conv2d-44            [-1, 512, 4, 4]       2,359,296
       BatchNorm2d-45            [-1, 512, 4, 4]           1,024
            Conv2d-46            [-1, 512, 4, 4]       2,359,296
       BatchNorm2d-47            [-1, 512, 4, 4]           1,024
        BasicBlock-48            [-1, 512, 4, 4]               0
            Linear-49                   [-1, 10]           5,130
    ================================================================
    Total params: 11,173,962
    Trainable params: 11,173,962
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 11.25
    Params size (MB): 42.63
    Estimated Total Size (MB): 53.89
    ----------------------------------------------------------------

## Output of image augmentation
![Albumentation](./images/augmentation.png)

## Training log
    Epoch 0: 100%|██████████| 704/704 [38:52<00:00,  0.30it/s, v_num=3, val_loss=2.310, val_acc=0.108]

## Misclassified Images
![Results](./images/results.png)

## Grad-CAM Images
![Grad-CAM](./images/grad_cam.png)

## EC2
Ran simple example as I don't have G type instance for EC2. 
![EC2](./images/EC2_example.png)

## Acknowledgments
This model is trained using repo listed below
* [pytorch_cifar_master](https://github.com/AkashDataScience/pytorch_cifar_master)
* [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)