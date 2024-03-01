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
h1 = w1*i1 + w2*i2  
h2 = w3*i1 + w4*i2  
a_h1 = σ(h1) = 1/(1+exp(-h1))  
a_h2 = σ(h2)  
o1 = w5*a_h1 + w6*a_h2  
o2 = w7*a_h1 + w8*w_h2  
a_o1 = σ(o1)  
a_o2 = σ(o2)  
E_total = E1 + E2  
E1 = (1/2) * (t1 - a_o1)<sup>2</sup>  
E2 = (1/2) * (t2 – a_o2)<sup>2</sup>  

#### Step 2: Backpropagate loss to hidden layer
For backpropagation, we will calculate partial derivative of total loss with respect to a weight in
hidden layer.
##### Partial derivative of total loss with respect to w5
∂E_Total/∂w5 = ∂(E1 + E2)/∂w5  
∂E_Total/∂w5 = ∂E1/∂w5  
∂E_Total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5  
∂E1/∂a_o1 = ∂((1/2) * (t1 – a_o1)2)/∂a_o1 = (a_o1 – t1)  
∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = a_o1*(1 – a_o1)  
∂o1/∂w5 = a_h1  
∂E_Total/∂w5 = (a_o1 – t1) * a_o1 * (1 – a_o1) * a_h1  

#### Step 3: Write partial derivative for weights of hidden layer
Based on formula derived in Step 2, we can write partial derivative of total loss with respect to 
each weights of hidden layer.
##### Partial derivative of total loss with respect to w5, w6, w7, w8
∂E_Total/∂w5 = (a_o1 – t1) * a_o1 * (1 – a_o1) * a_h1  
∂E_Total/∂w6 = (a_o1 – t1) * a_o1 * (1 – a_o1) * a_h2  
∂E_Total/∂w7 = (a_o2 – t2) * a_o2 * (1 – a_o2) * a_h1  
∂E_Total/∂w8 = (a_o2 – t2) * a_o2 * (1 – a_o2) * a_h2  

#### Step 4: Write partial derivative for activation in hidden layer
Now we have to calculate partial derivative of total loss with respect to activation in hidden
layer. This will be usefull while calculating partial derivative for weights of input layer.
##### Partial derivative of total loss with respect to a_h1, a_h2, 
∂E1/∂a_h1 = (a_o1 – t1)*a_o1*(1-a_o1)*w5  
∂E2/∂a_h1 = (a_o2 – t2)*a_o2*(1-a_o2)*w7  
∂E_total/∂a_h1 = (a_o1 – t1)*a_o1*(1-a_o1)*w5 +  (a_o2 – t2)*a_o2*(1-a_o2)*w7  
∂E_total/∂a_h2 = (a_o1 – t1)*a_o1*(1-a_o1)*w6 +  (a_o2 – t2)*a_o2*(1-a_o2)*w8  

#### Step 5: Write partial derivative for weights of input layer
Similar to hidden layer, we have to calculate partial derivative of total loss with respect to each
weights of input layer.
##### Partial derivative of total loss with respect to w1, w2, w3, w4
∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1  
∂E_total/∂w2 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2  
∂E_total/∂w3 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3  
∂E_total/∂w4 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w4  

#### Step 6: Final equations for input layer
Replacing values in equations shown in Step 5.
##### Partial derivative of total loss with respect to w1, w2, w3, w4
∂E_total/∂w1 = ((a_o1 – t1) * a_o1 * (1-a_o1) * w5 +  (a_o2 – t2) * a_o2 * (1-a_o2) * w7) * a_h1 * (1 - a_h1) * i1  
∂E_total/∂w2 = ((a_o1 – t1) * a_o1 * (1-a_o1) * w5 +  (a_o2 – t2) * a_o2 * (1-a_o2) * w7) * a_h1 * (1 - a_h1) * i2  
∂E_total/∂w3 = ((a_o1 – t1) * a_o1 * (1-a_o1) * w6 +  (a_o2 – t2) * a_o2 * (1-a_o2) * w8) * a_h2 * (1 - a_h2) * i1  
∂E_total/∂w4 = ((a_o1 – t1) * a_o1 * (1-a_o1) * w6 +  (a_o2 – t2) * a_o2 * (1-a_o2) * w8) * a_h2 * (1 - a_h2) * i2  

#### Step 7: Update weights
Based on values of partial derivative we will update our weights
new_w1 = w1 - ƞ * ∂E_total/∂w1  
new_w2 = w2 - ƞ * ∂E_total/∂w2  
new_w3 = w3 - ƞ * ∂E_total/∂w3  
new_w4 = w4 - ƞ * ∂E_total/∂w4  
new_w5 = w5 - ƞ * ∂E_total/∂w5  
new_w6 = w6 - ƞ * ∂E_total/∂w6  
new_w7 = w7 - ƞ * ∂E_total/∂w7  
new_w8 = w8 - ƞ * ∂E_total/∂w8  
where, ƞ is learning rate

#### Go to step 1

### Result of changing learning rate
Below plot shows loss of network with different learning rate [0.1, 0.2, 0.5, 0.8, 1, 2]

![Train Loss](Images/loss_with_different_lr.png)

As we can see in plot, as we increase learning rate model converges faster. This might we true in
this paeticular case. In general very small or very large learnig rate can have negative impact on 
training or model. 