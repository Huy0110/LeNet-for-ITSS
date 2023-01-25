import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import time
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        self.dilation = nn.Sequential(
            nn.Conv2d(1, 128, 15, dilation=4),
            # nn.Conv2d(32, 64, 15, dilation=4),
            # nn.Conv2d(64, 128, 15, dilation=4)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
	    
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        
        self.dense = nn.Sequential(
            nn.Linear(512, 120),
            nn.Dropout(p=0.75),
            nn.Linear(120, 84),
            nn.Linear(84, 12)
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 6, 3, stride=1, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(6, 16, 5, stride=1, padding=0),
        #     nn.MaxPool2d(2, 2)
        # )
        
        # self.fc = nn.Sequential(
        #     nn.Linear(400, 120),
        #     nn.Linear(120, 84),
        #     nn.Linear(84, 12)
        # )
        
        
    def forward(self, x):
        # out = self.conv(x)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        out = self.dilation(x)
        out = self.conv1(x)
        out = self.conv2(x)
        out = self.conv3(x)
        out = self.conv4(x)
        out = self.conv5(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        return out