import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        """
        Constructor for Net class. Create instance of layers required in model. 
        """
        super(Net, self).__init__()
        # First convolution layer with input channel = 1, output channel = 32, kernel size = 3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # Second convolution layer with input channel = 32, output channel = 64, kernel size = 3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Third convolution layer with input channel = 64, output channel = 128, kernel size = 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        # Forth convolution layer with input channel = 128, output channel = 256, kernel size = 3
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        # First fully connected layer with input feature = 4096 and output feature = 50
        self.fc1 = nn.Linear(4096, 50)
        # Second fully connected layer with input feature = 50 and output feature = 10
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """Method to define forward computation pass.

        Args:
            x (Tensor): Batch of images or single image

        Returns:
            Tensor: Tensor representing probability of each class
        """
        # n_in>n_out | r_in>r_out | j_in>j_out
        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4
        x = x.view(-1, 4096) # 4*4*256 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)