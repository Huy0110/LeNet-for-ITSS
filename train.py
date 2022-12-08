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
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from model import Lenet
from dataloader_mnist import trainloader, testloader, trainset, testset

learning_rate = 1e-3
batch_size = 64
epoches = 50

lenet = Lenet()
lenet.cuda()

criterian = nn.CrossEntropyLoss(size_average=False)
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)


def validation(i):
    testloss = 0.
    testacc = 0.
    lenet.eval()
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
    writer.add_scalar('Validation/loss', testloss, i+1)
    writer.add_scalar('Validation/acc', 100*testacc, i+1)
    lenet.train()

# train
for i in range(epoches):
    since = time.time()
    running_loss = 0.
    running_acc = 0.
    for (img, label) in trainloader:
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        
        optimizer.zero_grad()
        output = lenet(img)
        loss = criterian(output, label)
        # backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        # print('-------start----------')
        # print(loss.item())
        # print(correct_num.item())
        # print('-------stop-------')
        running_acc += correct_num.item()
    
    running_loss /= len(trainset)
    running_acc /= len(trainset)
    print("[%d/%d] Loss: %.5f, Acc: %.2f, Time: %.1f s" %(i+1, epoches, running_loss, 100*running_acc, time.time()-since))
    validation(i)
    writer.add_scalar('Trainning/loss', running_loss, i+1)
    writer.add_scalar('Trainning/acc', 100*running_acc, i+1)


