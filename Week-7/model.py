import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

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