import os.path
import torch


batchSizeTrain = 64
batchSizeTest = 1000
learningRate = 0.001
momentum = 0.5
weight_decay = 0.0
nEpochs = 3
layerDimensions = [512, 350, 100] # Move to after reading file
batchNorm = True
dataset = 'MNIST' # 'CIFAR10', 'MNIST', 'FashionMNIST'
dataPath = './data/'
saveDir = './results/'
os.path.isdir(saveDir) or os.mkdir(saveDir)
cudaFlag = torch.cuda.is_available()
