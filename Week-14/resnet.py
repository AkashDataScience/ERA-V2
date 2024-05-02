"""
ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import os
import torch
import utils
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(LightningModule):
    def __init__(self, block, num_blocks, num_classes=10, loss='cross_entropy', learning_rate=2e-4, momentum=0.9, optimizer="SGD",
                 epochs=20):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.loss = utils.get_criterion(loss)
        self.epochs = epochs

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = utils.get_optimizer(self, lr=self.learning_rate, momentum=self.momentum, optimizer_type="SGD")
        max_lr = utils.get_learning_rate(self, optimizer, self.loss, self.trainer.datamodule.train_dataloader())
        scheduler = utils.get_OneCycleLR_scheduler(optimizer, max_lr=max_lr,  epochs=self.epochs,
                                           steps_per_epoch=len(self.trainer.datamodule.train_dataloader()), max_at_epoch=0,
                                           anneal_strategy = 'linear', div_factor=10,
                                           final_div_factor=1)
        return [optimizer],[{"scheduler": scheduler, "interval": "step", "frequency": 1}]

def ResNet18(loss='cross_entropy', learning_rate=2e-4, momentum=0.9, optimizer="SGD", epochs=20):
    return ResNet(BasicBlock, [2, 2, 2, 2], loss=loss, learning_rate=learning_rate, momentum=momentum,
                  optimizer=optimizer, epochs=epochs)


def ResNet34(loss='cross_entropy', learning_rate=2e-4, momentum=0.9, optimizer="SGD", epochs=20):
    return ResNet(BasicBlock, [3, 4, 6, 3], loss=loss, learning_rate=learning_rate, momentum=momentum,
                  optimizer=optimizer, epochs=epochs)
