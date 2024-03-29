import torch.nn as nn
import torch.nn.functional as F

def _get_conv_block(in_channels, out_channels, kernel_size, stride, dilation, dropout_value):
    conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
                               nn.ReLU(),
                               nn.BatchNorm2d(out_channels),
                               nn.Dropout(dropout_value),
                               nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False),
                               nn.ReLU(),
                               nn.BatchNorm2d(out_channels),
                               nn.Dropout(dropout_value),
                               nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, stride=stride, dilation=dilation, bias=False),
                               nn.ReLU(),
                               nn.BatchNorm2d(out_channels),
                               nn.Dropout(dropout_value))
    return conv_block

def _get_depthwise_separable_conv_block(in_channels, out_channels, kernel_size, stride, dilation, dropout_value):
    conv_block = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, padding=1, bias=False, groups=in_channels),
                               nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                               nn.ReLU(),
                               nn.BatchNorm2d(out_channels),
                               nn.Dropout(dropout_value),
                               nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False),
                               nn.ReLU(),
                               nn.BatchNorm2d(out_channels),
                               nn.Dropout(dropout_value),
                               nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, stride=stride, dilation=dilation, bias=False),
                               nn.ReLU(),
                               nn.BatchNorm2d(out_channels),
                               nn.Dropout(dropout_value))
    return conv_block

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.conv_block_1 = _get_conv_block(3, 16, 3, 1, 2, 0.1)
        self.conv_block_2 = _get_conv_block(16, 32, 3, 2, 1, 0.1)
        self.conv_block_3 = _get_depthwise_separable_conv_block(32, 48, 3, 2, 1, 0.1)
        self.conv_block_4 = _get_conv_block(48, 64, 3, 1, 1, 0.1)
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=8))
        self.conv_block_5 = nn.Sequential(nn.Conv2d(64, 10, 1, bias=False))

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.gap(x)
        x = self.conv_block_5(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)