# Assignment
1. has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
2. total RF must be more than 44
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. use albumentation library and apply:
    1. horizontal flip
    2. shiftScaleRotate
    3. coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
8. make sure you're following code-modularity (else 0 for full assignment) 
9. upload to Github
10. Attempt S9-Assignment Solution.
11. Questions in the Assignment QnA are:
    1. copy and paste your model code from your model.py file (full code) [125]
    2. copy paste output of torch summary [125]
    3. copy-paste the code where you implemented albumentation transformation for all three transformations [125]
    4. copy paste your training log (you must be running validation/text after each Epoch [125]
    5. Share the link for your README.md file. [200]

# Introduction
The goal of this assignment is to create a model with 4 convolution blocks and use dilation and depth wise seperable convolution. 

# Output of image augmentation
![Albumentation](./Images/Augmentation.png)

# Model summary
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
             Conv2d-1           [-1, 16, 32, 32]             432
               ReLU-2           [-1, 16, 32, 32]               0
        BatchNorm2d-3           [-1, 16, 32, 32]              32
            Dropout-4           [-1, 16, 32, 32]               0
             Conv2d-5           [-1, 16, 32, 32]           2,304
               ReLU-6           [-1, 16, 32, 32]               0
        BatchNorm2d-7           [-1, 16, 32, 32]              32
            Dropout-8           [-1, 16, 32, 32]               0
             Conv2d-9           [-1, 16, 30, 30]           2,304
              ReLU-10           [-1, 16, 30, 30]               0
       BatchNorm2d-11           [-1, 16, 30, 30]              32
           Dropout-12           [-1, 16, 30, 30]               0
            Conv2d-13           [-1, 32, 30, 30]           4,608
              ReLU-14           [-1, 32, 30, 30]               0
       BatchNorm2d-15           [-1, 32, 30, 30]              64
           Dropout-16           [-1, 32, 30, 30]               0
            Conv2d-17           [-1, 32, 30, 30]           9,216
              ReLU-18           [-1, 32, 30, 30]               0
       BatchNorm2d-19           [-1, 32, 30, 30]              64
           Dropout-20           [-1, 32, 30, 30]               0
            Conv2d-21           [-1, 32, 15, 15]           9,216
              ReLU-22           [-1, 32, 15, 15]               0
       BatchNorm2d-23           [-1, 32, 15, 15]              64
           Dropout-24           [-1, 32, 15, 15]               0
            Conv2d-25           [-1, 32, 15, 15]             288
            Conv2d-26           [-1, 48, 15, 15]           1,536
              ReLU-27           [-1, 48, 15, 15]               0
       BatchNorm2d-28           [-1, 48, 15, 15]              96
           Dropout-29           [-1, 48, 15, 15]               0
            Conv2d-30           [-1, 48, 15, 15]          20,736
              ReLU-31           [-1, 48, 15, 15]               0
       BatchNorm2d-32           [-1, 48, 15, 15]              96
           Dropout-33           [-1, 48, 15, 15]               0
            Conv2d-34             [-1, 48, 8, 8]          20,736
              ReLU-35             [-1, 48, 8, 8]               0
       BatchNorm2d-36             [-1, 48, 8, 8]              96
           Dropout-37             [-1, 48, 8, 8]               0
            Conv2d-38             [-1, 64, 8, 8]          27,648
              ReLU-39             [-1, 64, 8, 8]               0
       BatchNorm2d-40             [-1, 64, 8, 8]             128
           Dropout-41             [-1, 64, 8, 8]               0
            Conv2d-42             [-1, 64, 8, 8]          36,864
              ReLU-43             [-1, 64, 8, 8]               0
       BatchNorm2d-44             [-1, 64, 8, 8]             128
           Dropout-45             [-1, 64, 8, 8]               0
            Conv2d-46             [-1, 64, 8, 8]          36,864
              ReLU-47             [-1, 64, 8, 8]               0
       BatchNorm2d-48             [-1, 64, 8, 8]             128
           Dropout-49             [-1, 64, 8, 8]               0
         AvgPool2d-50             [-1, 64, 1, 1]               0
            Conv2d-51             [-1, 10, 1, 1]             640
    ================================================================
    Total params: 174,352
    Trainable params: 174,352
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 4.60
    Params size (MB): 0.67
    Estimated Total Size (MB): 5.28
    ----------------------------------------------------------------

# Training log
    Epoch 1
    Train: Loss=1.6764 Batch_id=390 Accuracy=33.17: 100%|██████████| 391/391 [00:17<00:00, 22.71it/s]
    Test set: Average loss: 1.4971, Accuracy: 4379/10000 (43.79%)

    Epoch 2
    Train: Loss=1.2699 Batch_id=390 Accuracy=43.51: 100%|██████████| 391/391 [00:17<00:00, 22.89it/s]
    Test set: Average loss: 1.2706, Accuracy: 5367/10000 (53.67%)

    Epoch 3
    Train: Loss=1.2842 Batch_id=390 Accuracy=48.33: 100%|██████████| 391/391 [00:18<00:00, 20.85it/s]
    Test set: Average loss: 1.1633, Accuracy: 5830/10000 (58.30%)

    Epoch 4
    Train: Loss=1.3299 Batch_id=390 Accuracy=51.86: 100%|██████████| 391/391 [00:17<00:00, 22.92it/s]
    Test set: Average loss: 1.1164, Accuracy: 6032/10000 (60.32%)

    Epoch 5
    Train: Loss=1.2362 Batch_id=390 Accuracy=54.08: 100%|██████████| 391/391 [00:18<00:00, 21.33it/s]
    Test set: Average loss: 1.0227, Accuracy: 6248/10000 (62.48%)

    Epoch 6
    Train: Loss=1.1222 Batch_id=390 Accuracy=56.15: 100%|██████████| 391/391 [00:17<00:00, 22.93it/s]
    Test set: Average loss: 0.9852, Accuracy: 6502/10000 (65.02%)

    Epoch 7
    Train: Loss=1.2273 Batch_id=390 Accuracy=57.41: 100%|██████████| 391/391 [00:18<00:00, 21.60it/s]
    Test set: Average loss: 0.9325, Accuracy: 6681/10000 (66.81%)

    Epoch 8
    Train: Loss=1.1481 Batch_id=390 Accuracy=58.63: 100%|██████████| 391/391 [00:17<00:00, 22.86it/s]
    Test set: Average loss: 0.8900, Accuracy: 6872/10000 (68.72%)

    Epoch 9
    Train: Loss=1.1108 Batch_id=390 Accuracy=59.98: 100%|██████████| 391/391 [00:17<00:00, 21.96it/s]
    Test set: Average loss: 0.8730, Accuracy: 6884/10000 (68.84%)

    Epoch 10
    Train: Loss=1.1272 Batch_id=390 Accuracy=61.40: 100%|██████████| 391/391 [00:18<00:00, 21.51it/s]
    Test set: Average loss: 0.8192, Accuracy: 7095/10000 (70.95%)

    Epoch 11
    Train: Loss=1.1163 Batch_id=390 Accuracy=62.10: 100%|██████████| 391/391 [00:18<00:00, 21.67it/s]
    Test set: Average loss: 0.7814, Accuracy: 7279/10000 (72.79%)

    Epoch 12
    Train: Loss=0.9622 Batch_id=390 Accuracy=63.12: 100%|██████████| 391/391 [00:16<00:00, 23.12it/s]
    Test set: Average loss: 0.7465, Accuracy: 7398/10000 (73.98%)

    Epoch 13
    Train: Loss=1.0336 Batch_id=390 Accuracy=63.88: 100%|██████████| 391/391 [00:18<00:00, 21.43it/s]
    Test set: Average loss: 0.7687, Accuracy: 7310/10000 (73.10%)

    Epoch 14
    Train: Loss=0.9112 Batch_id=390 Accuracy=64.59: 100%|██████████| 391/391 [00:17<00:00, 22.92it/s]
    Test set: Average loss: 0.7250, Accuracy: 7511/10000 (75.11%)

    Epoch 15
    Train: Loss=1.3032 Batch_id=390 Accuracy=65.55: 100%|██████████| 391/391 [00:17<00:00, 22.47it/s]
    Test set: Average loss: 0.7035, Accuracy: 7587/10000 (75.87%)

    Epoch 16
    Train: Loss=1.0154 Batch_id=390 Accuracy=65.91: 100%|██████████| 391/391 [00:17<00:00, 22.69it/s]
    Test set: Average loss: 0.6974, Accuracy: 7564/10000 (75.64%)

    Epoch 17
    Train: Loss=0.8970 Batch_id=390 Accuracy=66.85: 100%|██████████| 391/391 [00:18<00:00, 21.35it/s]
    Test set: Average loss: 0.6814, Accuracy: 7632/10000 (76.32%)

    Epoch 18
    Train: Loss=0.9797 Batch_id=390 Accuracy=67.03: 100%|██████████| 391/391 [00:17<00:00, 22.58it/s]
    Test set: Average loss: 0.6497, Accuracy: 7761/10000 (77.61%)

    Epoch 19
    Train: Loss=0.9971 Batch_id=390 Accuracy=67.24: 100%|██████████| 391/391 [00:18<00:00, 21.48it/s]
    Test set: Average loss: 0.6449, Accuracy: 7762/10000 (77.62%)

    Epoch 20
    Train: Loss=0.8513 Batch_id=390 Accuracy=68.13: 100%|██████████| 391/391 [00:17<00:00, 22.76it/s]
    Test set: Average loss: 0.6451, Accuracy: 7795/10000 (77.95%)

    Epoch 21
    Train: Loss=0.8169 Batch_id=390 Accuracy=68.29: 100%|██████████| 391/391 [00:18<00:00, 21.68it/s]
    Test set: Average loss: 0.6252, Accuracy: 7860/10000 (78.60%)

    Epoch 22
    Train: Loss=0.8700 Batch_id=390 Accuracy=68.68: 100%|██████████| 391/391 [00:17<00:00, 22.98it/s]
    Test set: Average loss: 0.6191, Accuracy: 7890/10000 (78.90%)

    Epoch 23
    Train: Loss=0.7791 Batch_id=390 Accuracy=69.04: 100%|██████████| 391/391 [00:19<00:00, 20.24it/s]
    Test set: Average loss: 0.5987, Accuracy: 7893/10000 (78.93%)

    Epoch 24
    Train: Loss=1.1322 Batch_id=390 Accuracy=69.59: 100%|██████████| 391/391 [00:17<00:00, 22.91it/s]
    Test set: Average loss: 0.6154, Accuracy: 7869/10000 (78.69%)

    Epoch 25
    Train: Loss=0.6467 Batch_id=390 Accuracy=69.39: 100%|██████████| 391/391 [00:18<00:00, 21.68it/s]
    Test set: Average loss: 0.5976, Accuracy: 7971/10000 (79.71%)

    Epoch 26
    Train: Loss=0.9107 Batch_id=390 Accuracy=70.02: 100%|██████████| 391/391 [00:17<00:00, 22.43it/s]
    Test set: Average loss: 0.5933, Accuracy: 7982/10000 (79.82%)

    Epoch 27
    Train: Loss=1.0081 Batch_id=390 Accuracy=70.27: 100%|██████████| 391/391 [00:18<00:00, 21.71it/s]
    Test set: Average loss: 0.5518, Accuracy: 8142/10000 (81.42%)

    Epoch 28
    Train: Loss=0.8247 Batch_id=390 Accuracy=70.75: 100%|██████████| 391/391 [00:17<00:00, 22.50it/s]
    Test set: Average loss: 0.5725, Accuracy: 8046/10000 (80.46%)

    Epoch 29
    Train: Loss=0.6857 Batch_id=390 Accuracy=70.79: 100%|██████████| 391/391 [00:18<00:00, 21.23it/s]
    Test set: Average loss: 0.5721, Accuracy: 8015/10000 (80.15%)

    Epoch 30
    Train: Loss=0.6871 Batch_id=390 Accuracy=70.91: 100%|██████████| 391/391 [00:18<00:00, 21.57it/s]
    Test set: Average loss: 0.5647, Accuracy: 8055/10000 (80.55%)

    Epoch 31
    Train: Loss=0.8321 Batch_id=390 Accuracy=71.31: 100%|██████████| 391/391 [00:18<00:00, 20.94it/s]
    Test set: Average loss: 0.5523, Accuracy: 8066/10000 (80.66%)

    Epoch 32
    Train: Loss=0.7269 Batch_id=390 Accuracy=71.41: 100%|██████████| 391/391 [00:17<00:00, 22.40it/s]
    Test set: Average loss: 0.5343, Accuracy: 8143/10000 (81.43%)

    Epoch 33
    Train: Loss=0.9462 Batch_id=390 Accuracy=71.46: 100%|██████████| 391/391 [00:18<00:00, 21.16it/s]
    Test set: Average loss: 0.5306, Accuracy: 8177/10000 (81.77%)

    Epoch 34
    Train: Loss=0.9258 Batch_id=390 Accuracy=71.79: 100%|██████████| 391/391 [00:17<00:00, 22.33it/s]
    Test set: Average loss: 0.5255, Accuracy: 8194/10000 (81.94%)

    Epoch 35
    Train: Loss=0.8290 Batch_id=390 Accuracy=71.79: 100%|██████████| 391/391 [00:18<00:00, 20.92it/s]
    Test set: Average loss: 0.5221, Accuracy: 8248/10000 (82.48%)

    Epoch 36
    Train: Loss=0.8105 Batch_id=390 Accuracy=72.20: 100%|██████████| 391/391 [00:18<00:00, 20.85it/s]
    Test set: Average loss: 0.5219, Accuracy: 8219/10000 (82.19%)

    Epoch 37
    Train: Loss=0.8683 Batch_id=390 Accuracy=72.21: 100%|██████████| 391/391 [00:18<00:00, 21.40it/s]
    Test set: Average loss: 0.5159, Accuracy: 8222/10000 (82.22%)

    Epoch 38
    Train: Loss=0.7304 Batch_id=390 Accuracy=72.47: 100%|██████████| 391/391 [00:17<00:00, 22.32it/s]
    Test set: Average loss: 0.5207, Accuracy: 8244/10000 (82.44%)

    Epoch 39
    Train: Loss=0.6286 Batch_id=390 Accuracy=72.78: 100%|██████████| 391/391 [00:18<00:00, 21.09it/s]
    Test set: Average loss: 0.5118, Accuracy: 8273/10000 (82.73%)

    Epoch 40
    Train: Loss=0.8322 Batch_id=390 Accuracy=72.59: 100%|██████████| 391/391 [00:17<00:00, 22.30it/s]
    Test set: Average loss: 0.5013, Accuracy: 8287/10000 (82.87%)

    Epoch 41
    Train: Loss=0.7006 Batch_id=390 Accuracy=72.90: 100%|██████████| 391/391 [00:18<00:00, 20.82it/s]
    Test set: Average loss: 0.4994, Accuracy: 8286/10000 (82.86%)

    Epoch 42
    Train: Loss=0.9449 Batch_id=390 Accuracy=72.89: 100%|██████████| 391/391 [00:18<00:00, 21.17it/s]
    Test set: Average loss: 0.5108, Accuracy: 8275/10000 (82.75%)

    Epoch 43
    Train: Loss=0.4679 Batch_id=390 Accuracy=73.18: 100%|██████████| 391/391 [00:18<00:00, 20.82it/s]
    Test set: Average loss: 0.5149, Accuracy: 8281/10000 (82.81%)

    Epoch 44
    Train: Loss=0.8080 Batch_id=390 Accuracy=73.16: 100%|██████████| 391/391 [00:17<00:00, 22.28it/s]
    Test set: Average loss: 0.5070, Accuracy: 8276/10000 (82.76%)

    Epoch 45
    Train: Loss=0.8866 Batch_id=390 Accuracy=73.31: 100%|██████████| 391/391 [00:18<00:00, 21.22it/s]
    Test set: Average loss: 0.5167, Accuracy: 8264/10000 (82.64%)

    Epoch 46
    Train: Loss=0.6521 Batch_id=390 Accuracy=73.44: 100%|██████████| 391/391 [00:17<00:00, 21.94it/s]
    Test set: Average loss: 0.4939, Accuracy: 8289/10000 (82.89%)

    Epoch 47
    Train: Loss=0.9509 Batch_id=390 Accuracy=73.46: 100%|██████████| 391/391 [00:18<00:00, 21.43it/s]
    Test set: Average loss: 0.4981, Accuracy: 8325/10000 (83.25%)

    Epoch 48
    Train: Loss=0.7262 Batch_id=390 Accuracy=73.41: 100%|██████████| 391/391 [00:17<00:00, 22.36it/s]
    Test set: Average loss: 0.4884, Accuracy: 8366/10000 (83.66%)

    Epoch 49
    Train: Loss=0.6639 Batch_id=390 Accuracy=74.02: 100%|██████████| 391/391 [00:19<00:00, 20.04it/s]
    Test set: Average loss: 0.4926, Accuracy: 8345/10000 (83.45%)

    Epoch 50
    Train: Loss=0.6382 Batch_id=390 Accuracy=74.10: 100%|██████████| 391/391 [00:17<00:00, 21.97it/s]
    Test set: Average loss: 0.4811, Accuracy: 8379/10000 (83.79%)

# Performance Graphs
![Metrics](./Images/metrics.png)

# Misclassified Images
![LayerNorm](./Images/results.png)