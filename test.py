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
from model import Lenet
from dataloader_mnist import trainloader, testloader, trainset, testset

path_model = '/root/LeNet-for-ITSS/Checkponts/best.pth'

lenet = Lenet()
lenet.load_state_dict(torch.load(path_model))
lenet.cuda()
criterian = nn.CrossEntropyLoss()

lenet.eval()

testloss = 0.
testacc = 0.
for (img, label) in testloader:
    img = Variable(img).cuda()
    label = Variable(label).cuda()
    
    output = lenet(img)
    loss = criterian(output, label)
    testloss += loss.item()
    _, predict = torch.max(output, 1)
    num_correct = (predict == label).sum()
    testacc += num_correct.item()

testloss /= len(testset)
testacc /= len(testset)
print("Test: Loss: %.5f, Acc: %.2f %%" %(testloss, 100*testacc))
