import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from DataPrep import *
from Config import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inputDim = imgHeight*imgWidth*numChannels
        self.hiddenDim = [self.inputDim]+layerDimensions+[numClasses]
        self.actFunc = F.relu
        self.Layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.batchNorm = batchNorm
        self.drop = nn.Dropout(p=0)
        for i in range(len(self.hiddenDim) - 1):
            self.Layers.append(nn.Linear(self.hiddenDim[i], self.hiddenDim[i + 1]))
            if self.batchNorm:
                self.bns.append(nn.BatchNorm1d(self.hiddenDim[i+1]))

    def forward(self,x):
        x = x.reshape(-1,imgHeight*imgWidth*numChannels)
        for i in range(len(self.hiddenDim)-1):
            x = self.Layers[i](x)
            if self.batchNorm:
                x = self.bns[i](x)
            if i==len(self.hiddenDim)-2:
                break
            x = self.drop(x)
            x = self.actFunc(x)

        return F.log_softmax(x,dim=-1)

