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
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

path_model = '/root/LeNet-for-ITSS/Checkponts/best.pth'

lenet = Lenet()
lenet.load_state_dict(torch.load(path_model))
lenet.cuda()
criterian = nn.CrossEntropyLoss()

lenet.eval()

testloss = 0.
testacc = 0.
y_pred = []
y_true = []
for (img, label) in testloader:
    img = Variable(img).cuda()
    label = Variable(label).cuda()
    
    output = lenet(img)

    loss = criterian(output, label)
    testloss += loss.item()
    _, predict = torch.max(output, 1)
    num_correct = (predict == label).sum()
    testacc += num_correct.item()

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output) # Save Prediction
        
    label = label.data.cpu().numpy()
    y_true.extend(label)

testloss /= len(testset)
testacc /= len(testset)
print("Test: Loss: %.5f, Acc: %.2f %%" %(testloss, 100*testacc))

classes = ('Chat','Email','File','P2p','Streaming','Voip','Vpn_Chat','Vpn_Email','Vpn_File','Vpn_P2p','Vpn_Streaming','Vpn_Voip')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output_old.png')
