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
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from model import Lenet
from dataloader_mnist import trainloader, testloader, trainset, testset

writer = SummaryWriter()

parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', required=True, type=float)
parser.add_argument('--batch_size', required=True, type=int)
parser.add_argument('--epoches', required=True, type=int)
parser.add_argument('--opt', required=True, type=str)
parser.add_argument('--out_dir', required=True, type=str)

learning_rate = vars(parser.parse_args())['learning_rate']
batch_size = vars(parser.parse_args())['batch_size']
epoches = vars(parser.parse_args())['epoches']
opt = vars(parser.parse_args())['opt']
out_dir = vars(parser.parse_args())['out_dir']
maxValAcc = 0

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
# learning_rate = 1e-3
# batch_size = 64
# epoches = 100

lenet = Lenet()
print("Model: ", lenet)
lenet.cuda()

criterian = nn.CrossEntropyLoss()
if opt == 'SGD':
    optimizer = optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9)
elif opt == 'Adam':
    optimizer = optim.Adam(lenet.parameters(), lr=learning_rate)
else:
    printf("INVALID OPTIMIZER\n")
    printf("CHOSE ADAM OPTIMIZER\\n")
    optimizer = optim.Adam(lenet.parameters(), lr=learning_rate)



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
    return testacc

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
        running_acc += correct_num.item()
    
    path_save = os.path.join(out_dir, 'epoch_' + str(i) + '.pth')
    torch.save(lenet.state_dict(), path_save)
    running_loss /= len(trainset)
    running_acc /= len(trainset)
    print("[%d/%d] Loss: %.5f, Acc: %.2f, Time: %.1f s" %(i+1, epoches, running_loss, 100*running_acc, time.time()-since))
    testacc = validation(i)
    if testacc > maxValAcc :
        print("Save the best model at epoch %d with acc: %.2f \n" %(i+1, 100*testacc))
        path_save = os.path.join(out_dir, 'best.pth')
        torch.save(lenet.state_dict(), path_save)
    writer.add_scalar('Trainning/loss', running_loss, i+1)
    writer.add_scalar('Trainning/acc', 100*running_acc, i+1)


