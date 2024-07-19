# Assignment
## Part 1
<details>
<summary>Expand</summary>

1. Write your own UNet from scratch 
2. Train 4 times: 
    - MP+Tr+CE 
    - MP+Tr+Dice Loss 
    - StrConv+Tr+CE 
    - StrConv+Ups+Dice Loss 

## Introduction
The goal of this assignment is to implement Unet model from scratch for segmentation task. Train it
with different type of losses and layers.

## MP+Tr+CE
<details>
<summary>Expand</summary>

### Train logs
    Epoch 1
    Train: Loss=0.5826 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.46it/s]
    Test set: Average loss: 1.3338
    Epoch 2
    Train: Loss=0.4544 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.55it/s]
    Test set: Average loss: 0.8280
    Epoch 3
    Train: Loss=0.4951 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.51it/s]
    Test set: Average loss: 0.6460
    Epoch 4
    Train: Loss=0.4133 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.52it/s]
    Test set: Average loss: 1.1128
    Epoch 5
    Train: Loss=0.3978 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.48it/s]
    Test set: Average loss: 1.0972
    ...
    Epoch 21
    Train: Loss=0.1921 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.47it/s]
    Test set: Average loss: 0.9850
    Epoch 22
    Train: Loss=0.1754 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.51it/s]
    Test set: Average loss: 0.8336
    Epoch 23
    Train: Loss=0.1410 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.53it/s]
    Test set: Average loss: 0.6560
    Epoch 24
    Train: Loss=0.1979 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.52it/s]
    Test set: Average loss: 0.6963
    Epoch 25
    Train: Loss=0.1930 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.53it/s]
    Test set: Average loss: 0.5283

### Performance Graph
![MP+Tr+CE metrics](./images/MP_Tr_CE_metrics.png)

### Results
![MP+Tr+CE results](./images/MP_Tr_CE_results.png)

</details>

## MP+Tr+Dice Loss
<details>
<summary>Expand</summary>

### Train logs
    CUDA Available? True
    Epoch 1
    Train: Loss=0.1578 Batch_id=28: 100%|██████████| 29/29 [00:17<00:00,  1.70it/s]
    Test set: Average loss: 0.2559
    Epoch 2
    Train: Loss=0.1554 Batch_id=28: 100%|██████████| 29/29 [00:16<00:00,  1.74it/s]
    Test set: Average loss: 0.2501
    Epoch 3
    Train: Loss=0.1305 Batch_id=28: 100%|██████████| 29/29 [00:16<00:00,  1.73it/s]
    Test set: Average loss: 0.2115
    Epoch 4
    Train: Loss=0.1166 Batch_id=28: 100%|██████████| 29/29 [00:16<00:00,  1.71it/s]
    Test set: Average loss: 0.1905
    Epoch 5
    Train: Loss=0.1199 Batch_id=28: 100%|██████████| 29/29 [00:16<00:00,  1.74it/s]
    Test set: Average loss: 0.1816
    ...
    Epoch 21
    Train: Loss=0.0528 Batch_id=28: 100%|██████████| 29/29 [00:17<00:00,  1.70it/s]
    Test set: Average loss: 0.0948
    Epoch 22
    Train: Loss=0.0504 Batch_id=28: 100%|██████████| 29/29 [00:17<00:00,  1.69it/s]
    Test set: Average loss: 0.1257
    Epoch 23
    Train: Loss=0.0530 Batch_id=28: 100%|██████████| 29/29 [00:16<00:00,  1.72it/s]
    Test set: Average loss: 0.0909
    Epoch 24
    Train: Loss=0.0492 Batch_id=28: 100%|██████████| 29/29 [00:16<00:00,  1.74it/s]
    Test set: Average loss: 0.1136
    Epoch 25
    Train: Loss=0.0448 Batch_id=28: 100%|██████████| 29/29 [00:16<00:00,  1.74it/s]
    Test set: Average loss: 0.1120

### Performance Graph
![MP+Tr+Dice Loss metrics](./images/MP_Tr_Dice_Loss_metrics.png)

### Results
![MP+Tr+Dice Loss results](./images/MP_Tr_Dice_Loss_results.png)

</details>

## StrConv+Tr+CE
<details>
<summary>Expand</summary>

### Train logs
    CUDA Available? True
    Epoch 1
    Train: Loss=0.5796 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.42it/s]
    Test set: Average loss: 1.3183
    Epoch 2
    Train: Loss=0.5894 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.41it/s]
    Test set: Average loss: 0.5862
    Epoch 3
    Train: Loss=0.5051 Batch_id=57: 100%|██████████| 58/58 [00:17<00:00,  3.39it/s]
    Test set: Average loss: 0.5981
    Epoch 4
    Train: Loss=0.5162 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.43it/s]
    Test set: Average loss: 0.5238
    Epoch 5
    Train: Loss=0.4325 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.42it/s]
    Test set: Average loss: 0.5245
    ...
    Epoch 21
    Train: Loss=0.2533 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.47it/s]
    Test set: Average loss: 0.4071
    Epoch 22
    Train: Loss=0.2243 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.47it/s]
    Test set: Average loss: 0.3076
    Epoch 23
    Train: Loss=0.2697 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.48it/s]
    Test set: Average loss: 0.3325
    Epoch 24
    Train: Loss=0.2400 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.49it/s]
    Test set: Average loss: 0.4501
    Epoch 25
    Train: Loss=0.2470 Batch_id=57: 100%|██████████| 58/58 [00:16<00:00,  3.41it/s]
    Test set: Average loss: 0.2871

### Performance Graph
![StrConv+Tr+CE metrics](./images/StrConv_Tr_CE_metrics.png)

### Results
![StrConv+Tr+CE results](./images/StrConv_Tr_CE_results.png)

</details>

## StrConv+Ups+Dice Loss
<details>
<summary>Expand</summary>

### Train logs
    CUDA Available? True
    Epoch 1
    Train: Loss=0.1710 Batch_id=114: 100%|██████████| 115/115 [00:18<00:00,  6.26it/s]
    Test set: Average loss: 0.1596
    Epoch 2
    Train: Loss=0.1415 Batch_id=114: 100%|██████████| 115/115 [00:18<00:00,  6.31it/s]
    Test set: Average loss: 0.1661
    Epoch 3
    Train: Loss=0.1201 Batch_id=114: 100%|██████████| 115/115 [00:18<00:00,  6.29it/s]
    Test set: Average loss: 0.1223
    Epoch 4
    Train: Loss=0.0969 Batch_id=114: 100%|██████████| 115/115 [00:18<00:00,  6.28it/s]
    Test set: Average loss: 0.1148
    Epoch 5
    Train: Loss=0.0990 Batch_id=114: 100%|██████████| 115/115 [00:18<00:00,  6.28it/s]
    Test set: Average loss: 0.1261
    Epoch 20
    Train: Loss=0.0663 Batch_id=114: 100%|██████████| 115/115 [00:17<00:00,  6.42it/s]
    Test set: Average loss: 0.1391
    Epoch 21
    Train: Loss=0.0602 Batch_id=114: 100%|██████████| 115/115 [00:17<00:00,  6.43it/s]
    Test set: Average loss: 0.0642
    Epoch 22
    Train: Loss=0.0613 Batch_id=114: 100%|██████████| 115/115 [00:17<00:00,  6.41it/s]
    Test set: Average loss: 0.0651
    Epoch 23
    Train: Loss=0.0561 Batch_id=114: 100%|██████████| 115/115 [00:18<00:00,  6.34it/s]

### Performance Graph
![StrConv+Ups+Dice Loss metrics](./images/StrConv_Ups_Dice_Loss_metrics.png)

### Results
![StrConv+Ups+Dice Loss results](./images/StrConv_Ups_Dice_Loss_results.png)

</details>
</details>

## Part 2
<details>
<summary>Expand</summary>

1. Design a varition of a VAE to take image and it's label as input
2. Train 2 times:
    - Train on MNIST data
    - Train on CIFAR10 data
3. Generate 25 images from each model

## Introduction
The goal of this assignment is to implement VAE model from scratch for image generation task. Train it
with different public data.

## MNIST data
<details>
<summary>Expand</summary>

### train logs
    Epoch 29: 100% 1500/1500 [01:26<00:00, 17.27it/s, loss=-949, v_num=0]

### Results
![MNIST results](./images/mnist_results.png)

</details>

## CIFAR10 data
<details>
<summary>Expand</summary>

### train logs
    Epoch 29: 100% 1250/1250 [01:30<00:00, 13.80it/s, loss=-2.93e+03, v_num=1]

### Results
![CIFAR10 results](./images/CIFAR10_results.png)

</details>

## Acknowledgments
This model is trained using repo listed below
* [UNet](https://github.com/AkashDataScience/unet_pytorch)
* [VAE](https://github.com/AkashDataScience/vae_pytorch)