import torchvision
import torch
from Config import *
import os
import os.path
import numpy as np


os.path.isdir(dataPath) or os.mkdir(dataPath)
DataTrans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
if dataset == 'MNIST':
    datasetTrain = torchvision.datasets.MNIST(dataPath,train=True,download=True,
                                             transform=DataTrans)
    datasetTest = torchvision.datasets.MNIST(dataPath, train=True, download=True,
                                             transform=DataTrans)
    imgHeight, imgWidth = datasetTest.data.shape[1:]
    numChannels = 1
    numClasses = np.asscalar(torch.max(datasetTest.targets[:]) + 1)
if dataset == 'FashionMNIST':
    datasetTrain = torchvision.datasets.FashionMNIST(dataPath, train=True, download=True,
                                              transform=DataTrans)
    datasetTest = torchvision.datasets.FashionMNIST(dataPath, train=True, download=True,
                                             transform=DataTrans)
    imgHeight, imgWidth = datasetTest.data.shape[1:]
    numChannels = 1
    numClasses = np.asscalar(torch.max(datasetTest.targets[:]) + 1)
if dataset == 'CIFAR10':
    datasetTrain = torchvision.datasets.CIFAR10(dataPath, train=True, download=True,
                                              transform=DataTrans)
    datasetTest = torchvision.datasets.CIFAR10(dataPath, train=True, download=True,
                                             transform=DataTrans)
    imgHeight, imgWidth, numChannels = datasetTest.data.shape[1:]
    numClasses = np.asscalar(torch.max(datasetTest.targets[:]) + 1)

trainLoader = torch.utils.data.DataLoader(datasetTrain, batch_size=batchSizeTrain, shuffle=True)
testLoader = torch.utils.data.DataLoader(datasetTest, batch_size=batchSizeTest, shuffle=True)
