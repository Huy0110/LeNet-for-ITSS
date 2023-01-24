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

# from mnist import MNIST as MN2

# mndata = MN2('FlowAllLayers')

# images, labels = mndata.load_testing()
# print("Label: ",labels)
# print("Images: ", images)
# or
# images, labels = mndata.load_testing()


trans_img = transforms.ToTensor()
trainset = MNIST('./FlowAllLayers', train=True, transform=trans_img)
testset = MNIST('./FlowAllLayers', train=False, transform=trans_img)

print("test set: ", testset)

batch_size = 64

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)