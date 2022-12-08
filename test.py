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
from dataloader_mnist import trainloader, testloader

trans_img = transforms.ToTensor()
lenet = Lenet()
lenet.cuda()

criterian = nn.CrossEntropyLoss(size_average=False)
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)


lenet.eval()

testloss = 0.
testacc = 0.
for (img, label) in testloader:
    img = Variable(img).cuda()
    label = Variable(label).cuda()
    
    output = lenet(img)
    loss = criterian(output, label)
    testloss += loss.data[0]
    _, predict = torch.max(output, 1)
    num_correct = (predict == label).sum()
    testacc += num_correct.data[0]

testloss /= len(testset)
testacc /= len(testset)
print("Test: Loss: %.5f, Acc: %.2f %%" %(testloss, 100*testacc))
