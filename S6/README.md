# Session 6 - Backpropagatino and Advanced Architecture
A step by step guide to backpropagation and brief introduction to architectures like VGG and ResNet. 

## Part 1 Backpropagation
### Screenshot of excel sheel
![Backpropagation](Images/backpropagation.png)
### Steps for backpropagation
Let's consider network shown in image below. 

![Network](Images/Network.png)

#### Step 1: Forward pass
First we pass input values through neural network and get predition. Then we calculate loss between 
actual values and predicted values. 
##### Equations for Forwatd pass
h1 = w1\*i1 + w2\*i2  
h2 = w3\*i1 + w4\*i2  
a_h1 = σ(h1) = 1/(1+exp(-h1))  
a_h2 = σ(h2)  
o1 = w5\*a_h1 + w6\*a_h2  
o2 = w7\*a_h1 + w8\*w_h2  
a_o1 = σ(o1)  
a_o2 = σ(o2)  
E_total = E1 + E2  
E1 = (1/2) \* (t1 - a_o1)<sup>2</sup>  
E2 = (1/2) \* (t2 – a_o2)<sup>2</sup>  

#### Step 2: Backpropagate loss to hidden layer
For backpropagation, we will calculate partial derivative of total loss with respect to a weight in
hidden layer.
##### Partial derivative of total loss with respect to w5
∂E_Total/∂w5 = ∂(E1 + E2)/∂w5  
∂E_Total/∂w5 = ∂E1/∂w5  
∂E_Total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1\*∂a_o1/∂o1\*∂o1/∂w5  
∂E1/∂a_o1 = ∂((1/2) \* (t1 – a_o1)2)/∂a_o1 = (a_o1 – t1)  
∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = a_o1\*(1 – a_o1)  
∂o1/∂w5 = a_h1  
∂E_Total/∂w5 = (a_o1 – t1) \* a_o1 \* (1 – a_o1) \* a_h1  

#### Step 3: Write partial derivative for weights of hidden layer
Based on formula derived in Step 2, we can write partial derivative of total loss with respect to 
each weights of hidden layer.
##### Partial derivative of total loss with respect to w5, w6, w7, w8
∂E_Total/∂w5 = (a_o1 – t1) \* a_o1 \* (1 – a_o1) \* a_h1  
∂E_Total/∂w6 = (a_o1 – t1) \* a_o1 \* (1 – a_o1) \* a_h2  
∂E_Total/∂w7 = (a_o2 – t2) \* a_o2 \* (1 – a_o2) \* a_h1  
∂E_Total/∂w8 = (a_o2 – t2) \* a_o2 \* (1 – a_o2) \* a_h2  

#### Step 4: Write partial derivative for activation in hidden layer
Now we have to calculate partial derivative of total loss with respect to activation in hidden
layer. This will be usefull while calculating partial derivative for weights of input layer.
##### Partial derivative of total loss with respect to a_h1, a_h2, 
∂E1/∂a_h1 = (a_o1 – t1)\*a_o1\*(1-a_o1)\*w5  
∂E2/∂a_h1 = (a_o2 – t2)\*a_o2\*(1-a_o2)\*w7  
∂E_total/∂a_h1 = (a_o1 – t1)\*a_o1\*(1-a_o1)\*w5 +  (a_o2 – t2)\*a_o2\*(1-a_o2)\*w7  
∂E_total/∂a_h2 = (a_o1 – t1)\*a_o1\*(1-a_o1)\*w6 +  (a_o2 – t2)\*a_o2\*(1-a_o2)\*w8  

#### Step 5: Write partial derivative for weights of input layer
Similar to hidden layer, we have to calculate partial derivative of total loss with respect to each
weights of input layer.
##### Partial derivative of total loss with respect to w1, w2, w3, w4
∂E_total/∂w1 = ∂E_total/∂a_h1 \* ∂a_h1/∂h1 \* ∂h1/∂w1  
∂E_total/∂w2 = ∂E_total/∂a_h1 \* ∂a_h1/∂h1 \* ∂h1/∂w2  
∂E_total/∂w3 = ∂E_total/∂a_h2 \* ∂a_h2/∂h2 \* ∂h2/∂w3  
∂E_total/∂w4 = ∂E_total/∂a_h2 \* ∂a_h2/∂h2 \* ∂h2/∂w4  

#### Step 6: Final equations for input layer
Replacing values in equations shown in Step 5.
##### Partial derivative of total loss with respect to w1, w2, w3, w4
∂E_total/∂w1 = ((a_o1 – t1) \* a_o1 \* (1-a_o1) \* w5 +  (a_o2 – t2) \* a_o2 \* (1-a_o2) \* w7) \* a_h1 \* (1 - a_h1) \* i1  
∂E_total/∂w2 = ((a_o1 – t1) \* a_o1 \* (1-a_o1) \* w5 +  (a_o2 – t2) \* a_o2 \* (1-a_o2) \* w7) \* a_h1 \* (1 - a_h1) \* i2  
∂E_total/∂w3 = ((a_o1 – t1) \* a_o1 \* (1-a_o1) \* w6 +  (a_o2 – t2) \* a_o2 \* (1-a_o2) \* w8) \* a_h2 \* (1 - a_h2) \* i1  
∂E_total/∂w4 = ((a_o1 – t1) \* a_o1 \* (1-a_o1) \* w6 +  (a_o2 – t2) \* a_o2 \* (1-a_o2) \* w8) \* a_h2 \* (1 - a_h2) \* i2  

#### Step 7: Update weights
Based on values of partial derivative we will update our weights
new_w1 = w1 - ƞ \* ∂E_total/∂w1  
new_w2 = w2 - ƞ \* ∂E_total/∂w2  
new_w3 = w3 - ƞ \* ∂E_total/∂w3  
new_w4 = w4 - ƞ \* ∂E_total/∂w4  
new_w5 = w5 - ƞ \* ∂E_total/∂w5  
new_w6 = w6 - ƞ \* ∂E_total/∂w6  
new_w7 = w7 - ƞ \* ∂E_total/∂w7  
new_w8 = w8 - ƞ \* ∂E_total/∂w8  
where, ƞ is learning rate

#### Go to step 1

### Result of changing learning rate
Below plot shows loss of network with different learning rate [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]

![Train Loss](Images/loss_with_different_lr.png)

As we can see in plot, as we increase learning rate model converges faster. This might we true in
this paeticular case. In general very small or very large learnig rate can have negative impact on 
training or model. 

## Part 2 Train neural network

1. Validation accuracy: 99.41%.
2. Total parameters: 12,442
3. Epoch: 12

### Architecture

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 28, 28]              80
        BatchNorm2d-2            [-1, 8, 28, 28]              16
            Dropout-3            [-1, 8, 28, 28]               0
                Conv2d-4            [-1, 8, 28, 28]             584
        BatchNorm2d-5            [-1, 8, 28, 28]              16
            Dropout-6            [-1, 8, 28, 28]               0
                Conv2d-7            [-1, 8, 28, 28]             584
        BatchNorm2d-8            [-1, 8, 28, 28]              16
            MaxPool2d-9            [-1, 8, 14, 14]               0
            Dropout-10            [-1, 8, 14, 14]               0
            Conv2d-11           [-1, 16, 14, 14]           1,168
        BatchNorm2d-12           [-1, 16, 14, 14]              32
            Dropout-13           [-1, 16, 14, 14]               0
            Conv2d-14           [-1, 16, 12, 12]           2,320
        BatchNorm2d-15           [-1, 16, 12, 12]              32
            MaxPool2d-16             [-1, 16, 6, 6]               0
            Dropout-17             [-1, 16, 6, 6]               0
            Conv2d-18             [-1, 32, 4, 4]           4,640
        BatchNorm2d-19             [-1, 32, 4, 4]              64
            Dropout-20             [-1, 32, 4, 4]               0
            Conv2d-21             [-1, 10, 2, 2]           2,890
    ================================================================
    Total params: 12,442
    Trainable params: 12,442
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.53
    Params size (MB): 0.05
    Estimated Total Size (MB): 0.58
    ----------------------------------------------------------------

### Training Stats

    epoch=1 loss=0.13055013120174408 batch_id=468: 100%|██████████| 469/469 [01:15<00:00,  6.22it/s]

    Test set: Average loss: 0.1237, Accuracy: 9639/10000 (96.39%)

    Adjusting learning rate of group 0 to 1.0000e-02.
    epoch=2 loss=0.07950275391340256 batch_id=468: 100%|██████████| 469/469 [01:09<00:00,  6.71it/s]

    Test set: Average loss: 0.0492, Accuracy: 9849/10000 (98.49%)

    Adjusting learning rate of group 0 to 1.0000e-02.
    epoch=3 loss=0.08174591511487961 batch_id=468: 100%|██████████| 469/469 [01:08<00:00,  6.85it/s]

    Test set: Average loss: 0.0435, Accuracy: 9863/10000 (98.63%)

    Adjusting learning rate of group 0 to 1.0000e-02.
    epoch=4 loss=0.024415865540504456 batch_id=468: 100%|██████████| 469/469 [01:08<00:00,  6.88it/s]

    Test set: Average loss: 0.0345, Accuracy: 9891/10000 (98.91%)

    Adjusting learning rate of group 0 to 1.0000e-02.
    epoch=5 loss=0.057230088859796524 batch_id=468: 100%|██████████| 469/469 [01:10<00:00,  6.67it/s]

    Test set: Average loss: 0.0355, Accuracy: 9888/10000 (98.88%)

    Adjusting learning rate of group 0 to 1.0000e-02.
    epoch=6 loss=0.06492535769939423 batch_id=468: 100%|██████████| 469/469 [01:10<00:00,  6.64it/s]

    Test set: Average loss: 0.0299, Accuracy: 9908/10000 (99.08%)

    Adjusting learning rate of group 0 to 1.0000e-02.
    epoch=7 loss=0.05206746235489845 batch_id=468: 100%|██████████| 469/469 [01:10<00:00,  6.66it/s]

    Test set: Average loss: 0.0298, Accuracy: 9911/10000 (99.11%)

    Adjusting learning rate of group 0 to 1.0000e-02.
    epoch=8 loss=0.010062999092042446 batch_id=468: 100%|██████████| 469/469 [01:08<00:00,  6.87it/s]

    Test set: Average loss: 0.0282, Accuracy: 9912/10000 (99.12%)

    Adjusting learning rate of group 0 to 1.0000e-02.
    epoch=9 loss=0.01832321658730507 batch_id=468: 100%|██████████| 469/469 [01:10<00:00,  6.66it/s]

    Test set: Average loss: 0.0235, Accuracy: 9930/10000 (99.30%)

    Adjusting learning rate of group 0 to 1.0000e-02.
    epoch=10 loss=0.037380460649728775 batch_id=468: 100%|██████████| 469/469 [01:09<00:00,  6.76it/s]

    Test set: Average loss: 0.0233, Accuracy: 9926/10000 (99.26%)

    Adjusting learning rate of group 0 to 1.0000e-03.
    epoch=11 loss=0.06139003112912178 batch_id=468: 100%|██████████| 469/469 [01:09<00:00,  6.71it/s]

    Test set: Average loss: 0.0211, Accuracy: 9936/10000 (99.36%)

    Adjusting learning rate of group 0 to 1.0000e-03.
    epoch=12 loss=0.006513600703328848 batch_id=468: 100%|██████████| 469/469 [01:09<00:00,  6.76it/s]

    Test set: Average loss: 0.0208, Accuracy: 9941/10000 (99.41%)

    Adjusting learning rate of group 0 to 1.0000e-03.