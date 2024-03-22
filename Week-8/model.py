import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1
class Model_BN(nn.Module):
    def __init__(self):
        super(Model_BN, self).__init__()                                                 # Input  --> Output     | Rin --> Rout
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3, 16, 3, bias=False, padding=1),    # 32x32x3 --> 32x32x16   | 1   --> 3
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(16, 16, 3, bias=False, padding=1),   # 32x32x16 --> 32x32x16  | 3   --> 5
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value))

        self.tran_block_1 = nn.Sequential(nn.Conv2d(16, 8, 1, bias=False),              # 32x32x16--> 32x32x8   | 5   --> 5
                                          nn.MaxPool2d(2, 2))                           # 32x32x8 --> 16x16x8   | 5   --> 6

        self.conv_block_2 = nn.Sequential(nn.Conv2d(8, 16, 3, bias=False, padding=1),     # 16x16x8 --> 16x16x16   | 6   --> 10
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(16, 16, 3, bias=False, padding=1),   # 16x16x16 --> 16x16x16  | 10  --> 14
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(16, 16, 3, bias=False, padding=1),  # 16x16x16--> 16x16x16  | 14  --> 18
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value))

        self.tran_block_2 = nn.Sequential(nn.Conv2d(16, 8, 1, bias=False),              # 16x16x16--> 16x16x8   | 18  --> 18
                                          nn.MaxPool2d(2, 2))                           # 16x16x8 --> 8x8x8     | 18  --> 20

        self.conv_block_3 = nn.Sequential(nn.Conv2d(8, 16, 3, bias=False, padding=1),    # 8x8x8   --> 8x8x16     | 20  --> 28
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(16, 16, 3, bias=False, padding=1),   # 8x8x16   --> 8x8x16    | 28  --> 36
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(16, 16, 3, bias=False),             # 8x8x16 --> 6x6x16     | 36  --> 44
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value))

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))

        self.conv_block_4 = nn.Sequential(nn.Conv2d(16, 10, 1, bias=False))

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.tran_block_1(x)
        x = self.conv_block_2(x)
        x = self.tran_block_2(x)
        x = self.conv_block_3(x)
        x = self.gap(x)
        x = self.conv_block_4(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
    
class Model_GN(nn.Module):
    def __init__(self):
        super(Model_GN, self).__init__()                                                 # Input  --> Output     | Rin --> Rout
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3, 16, 3, bias=False, padding=1),    # 32x32x3 --> 32x32x16   | 1   --> 3
                                          nn.GroupNorm(4, 16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(16, 16, 3, bias=False, padding=1),   # 32x32x16 --> 32x32x16  | 3   --> 5
                                          nn.GroupNorm(4, 16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value))

        self.tran_block_1 = nn.Sequential(nn.Conv2d(16, 8, 1, bias=False),              # 32x32x16--> 32x32x8   | 5   --> 5
                                          nn.MaxPool2d(2, 2))                           # 32x32x8 --> 16x16x8   | 5   --> 6

        self.conv_block_2 = nn.Sequential(nn.Conv2d(8, 32, 3, bias=False, padding=1),     # 16x16x8 --> 16x16x32   | 6   --> 10
                                          nn.GroupNorm(4, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(32, 32, 3, bias=False, padding=1),   # 16x16x32 --> 16x16x32  | 10  --> 14
                                          nn.GroupNorm(4, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(32, 32, 3, bias=False, padding=1),  # 16x16x32--> 16x16x32  | 14  --> 18
                                          nn.GroupNorm(4, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value))

        self.tran_block_2 = nn.Sequential(nn.Conv2d(32, 8, 1, bias=False),              # 16x16x32--> 16x16x8   | 18  --> 18
                                          nn.MaxPool2d(2, 2))                           # 16x16x8 --> 8x8x8     | 18  --> 20

        self.conv_block_3 = nn.Sequential(nn.Conv2d(8, 32, 3, bias=False, padding=1),    # 8x8x8   --> 8x8x32     | 20  --> 28
                                          nn.GroupNorm(4, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(32, 32, 3, bias=False, padding=1),   # 8x8x32   --> 8x8x32    | 28  --> 36
                                          nn.GroupNorm(4, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(32, 32, 3, bias=False),             # 8x8x32 --> 6x6x32     | 36  --> 44
                                          nn.GroupNorm(4, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value))

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))

        self.conv_block_4 = nn.Sequential(nn.Conv2d(32, 10, 1, bias=False))

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.tran_block_1(x)
        x = self.conv_block_2(x)
        x = self.tran_block_2(x)
        x = self.conv_block_3(x)
        x = self.gap(x)
        x = self.conv_block_4(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
    
class Model_LN(nn.Module):
    def __init__(self):
        super(Model_LN, self).__init__()                                                 # Input  --> Output     | Rin --> Rout
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3, 16, 3, bias=False, padding=1),    # 32x32x3 --> 32x32x16   | 1   --> 3
                                          nn.GroupNorm(1, 16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(16, 16, 3, bias=False, padding=1),   # 32x32x16 --> 32x32x16  | 3   --> 5
                                          nn.GroupNorm(1, 16),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value))

        self.tran_block_1 = nn.Sequential(nn.Conv2d(16, 8, 1, bias=False),              # 32x32x16--> 32x32x8   | 5   --> 5
                                          nn.MaxPool2d(2, 2))                           # 32x32x8 --> 16x16x8   | 5   --> 6

        self.conv_block_2 = nn.Sequential(nn.Conv2d(8, 32, 3, bias=False, padding=1),     # 16x16x8 --> 16x16x32   | 6   --> 10
                                          nn.GroupNorm(1, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(32, 32, 3, bias=False, padding=1),   # 16x16x32 --> 16x16x32  | 10  --> 14
                                          nn.GroupNorm(1, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(32, 32, 3, bias=False, padding=1),  # 16x16x32--> 16x16x32  | 14  --> 18
                                          nn.GroupNorm(1, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value))

        self.tran_block_2 = nn.Sequential(nn.Conv2d(32, 8, 1, bias=False),              # 16x16x32--> 16x16x8   | 18  --> 18
                                          nn.MaxPool2d(2, 2))                           # 16x16x8 --> 8x8x8     | 18  --> 20

        self.conv_block_3 = nn.Sequential(nn.Conv2d(8, 32, 3, bias=False, padding=1),    # 8x8x8   --> 8x8x32     | 20  --> 28
                                          nn.GroupNorm(1, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(32, 32, 3, bias=False, padding=1),   # 8x8x32   --> 8x8x32    | 12  --> 16
                                          nn.GroupNorm(1, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(32, 32, 3, bias=False),             # 8x8x32 --> 6x6x32     | 16  --> 20
                                          nn.GroupNorm(1, 32),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_value))

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))

        self.conv_block_4 = nn.Sequential(nn.Conv2d(32, 10, 1, bias=False))

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.tran_block_1(x)
        x = self.conv_block_2(x)
        x = self.tran_block_2(x)
        x = self.conv_block_3(x)
        x = self.gap(x)
        x = self.conv_block_4(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

########################################
############# Session-7 ################
########################################

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()                                      # Input  --> Output     | Rin --> Rout
        self.conv_block_1 = nn.Sequential(nn.Conv2d(1, 4, 3, bias=False),    # 28x28x1 --> 28x28x4   | 1   --> 3
                                          nn.ReLU(),
                                          nn.BatchNorm2d(4),
                                          nn.Conv2d(4, 8, 3, bias=False),    # 28x28x4 --> 28x28x8   | 3   --> 5
                                          nn.ReLU(),
                                          nn.BatchNorm2d(8),
                                          nn.Conv2d(8, 16, 3, bias=False),   # 28x28x8 --> 28x28x16  | 5   --> 7
                                          nn.ReLU(),
                                          nn.BatchNorm2d(16))

        self.tran_block_1 = nn.Sequential(nn.Conv2d(16, 4, 1, bias=False),   # 28x28x16--> 28x28x4   | 7   --> 7
                                          nn.MaxPool2d(2, 2))                # 28x28x4 --> 14x14x4   | 7   --> 8

        self.conv_block_2 = nn.Sequential(nn.Conv2d(4, 4, 3, bias=False),    # 14x14x4 --> 14x14x4   | 8   --> 12
                                          nn.ReLU(),
                                          nn.BatchNorm2d(4),
                                          nn.Conv2d(4, 8, 3, bias=False),    # 14x14x4 --> 12x12x8   | 12  --> 16
                                          nn.ReLU(),
                                          nn.BatchNorm2d(8),
                                          nn.Conv2d(8, 12, 3, bias=False),   # 12x12x8 --> 10x10x16  | 16  --> 20
                                          nn.ReLU(),
                                          nn.BatchNorm2d(12),
                                          nn.Conv2d(12, 16, 3, bias=False),  # 5x5x4    --> 3x3x8    | 22   --> 30
                                          nn.ReLU(),
                                          nn.BatchNorm2d(16),
                                          nn.Conv2d(16, 10, 3, bias=False))  # 3x3x8    --> 1x1x10   | 30   --> 38

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.tran_block_1(x)
        x = self.conv_block_2(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
    
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()                                      # Input  --> Output     | Rin --> Rout
        self.conv_block_1 = nn.Sequential(nn.Conv2d(1, 10, 3, bias=False),   # 28x28x1 --> 26x26x4   | 1   --> 3
                                          nn.ReLU(),
                                          nn.BatchNorm2d(10),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(10, 10, 3, bias=False),  # 26x26x4 --> 24x24x8   | 3   --> 5
                                          nn.ReLU(),
                                          nn.BatchNorm2d(10),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(10, 20, 3, bias=False),  # 24x24x8 --> 22x22x16   | 5   --> 7
                                          nn.ReLU(),
                                          nn.BatchNorm2d(20),
                                          nn.Dropout(dropout_value))

        self.tran_block_1 = nn.Sequential(nn.Conv2d(20, 5, 1, bias=False),   # 22x22x16--> 22x22x4   | 7   --> 7
                                          nn.MaxPool2d(2, 2))                # 11x11x4 --> 11x11x4   | 7   --> 8

        self.conv_block_2 = nn.Sequential(nn.Conv2d(5, 10, 3, bias=False),   # 11x11x4 --> 9x9x4     | 8   --> 12
                                          nn.ReLU(),
                                          nn.BatchNorm2d(10),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(10, 15, 3, bias=False),  # 9x9x4 --> 7x7x8       | 12  --> 16
                                          nn.ReLU(),
                                          nn.BatchNorm2d(15),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(15, 20, 3, bias=False),  # 7x7x8 --> 5x5x16      | 16  --> 20
                                          nn.ReLU(),
                                          nn.BatchNorm2d(20),
                                          nn.Dropout(dropout_value))

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=5))

        self.conv_block_3 = nn.Sequential(nn.Conv2d(20, 10, 1, bias=False))

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.tran_block_1(x)
        x = self.conv_block_2(x)
        x = self.gap(x)
        x = self.conv_block_3(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
    
class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()                                      # Input  --> Output     | Rin --> Rout
        self.conv_block_1 = nn.Sequential(nn.Conv2d(1, 10, 3, bias=False),   # 28x28x1 --> 26x26x4   | 1   --> 3
                                          nn.ReLU(),
                                          nn.BatchNorm2d(10),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(10, 10, 3, bias=False),  # 26x26x4 --> 24x24x8   | 3   --> 5
                                          nn.ReLU(),
                                          nn.BatchNorm2d(10),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(10, 20, 3, bias=False),  # 24x24x8 --> 22x22x16  | 5   --> 7
                                          nn.ReLU(),
                                          nn.BatchNorm2d(20),
                                          nn.Dropout(dropout_value))

        self.tran_block_1 = nn.Sequential(nn.Conv2d(20, 5, 1, bias=False),   # 22x22x16--> 22x22x4   | 7   --> 7
                                          nn.MaxPool2d(2, 2))                # 11x11x4 --> 11x11x4   | 7   --> 8

        self.conv_block_2 = nn.Sequential(nn.Conv2d(5, 10, 3, bias=False),   # 11x11x4 --> 9x9x4     | 8   --> 12
                                          nn.ReLU(),
                                          nn.BatchNorm2d(10),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(10, 15, 3, bias=False),  # 9x9x4 --> 7x7x8       | 12  --> 16
                                          nn.ReLU(),
                                          nn.BatchNorm2d(15),
                                          nn.Dropout(dropout_value),
                                          nn.Conv2d(15, 20, 3, bias=False),  # 7x7x8 --> 5x5x16      | 16  --> 20
                                          nn.ReLU(),
                                          nn.BatchNorm2d(20),
                                          nn.Dropout(dropout_value))

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=5))

        self.conv_block_3 = nn.Sequential(nn.Conv2d(20, 10, 1, bias=False))

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.tran_block_1(x)
        x = self.conv_block_2(x)
        x = self.gap(x)
        x = self.conv_block_3(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

#######################################
############## Session-6 ##############
#######################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.1)              # Input  --> Output     | Rin --> Rout
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28x1 --> 28x28x8   | 1   --> 3 
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)  # 28x28x8 --> 28x28x8   | 3   --> 5
        self.conv2_bn = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)  # 28x28x8 --> 28x28x8   | 5   --> 7
        self.conv3_bn = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)             # 28x28x8 --> 14x14x8   | 7   --> 8
        self.conv4 = nn.Conv2d(8, 16, 3, padding=1) # 14x14x8 --> 14x14x16  | 8   --> 12
        self.conv4_bn = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 3)           # 14x14x16 --> 12x12x16 | 12  --> 16
        self.conv5_bn = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)             # 12x12x16 --> 6x6x16   | 16  --> 18 
        self.conv6 = nn.Conv2d(16, 32, 3)           # 6x6x16   --> 4x4x32   | 18  --> 26
        self.conv6_bn = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 10, 3)           # 4x4x32   --> 2x2x10   | 26  --> 34

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = self.dropout(x)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = self.dropout(x)
        x = self.conv7(x)
        x = x.mean([2, 3])
        x = x.view(-1, 10)
        return F.log_softmax(x)