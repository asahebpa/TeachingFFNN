import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Config import *
from DataPrep import *

def train(model,optimizer):
    model.train()
    if cudaFlag:
        network = model.cuda()
    for batchIdx, (data, target) in enumerate(trainLoader):
        # move to cuda
        if cudaFlag:
            target = target.cuda()
            data = data.cuda()
        output = network(data)
        loss = F.nll_loss(output, target)
        l1_penalty = torch.nn.L1Loss(reduction='mean')
        reg_loss = 0
        for param in model.parameters():
            reg_loss += l1_penalty(param,torch.zeros_like(param))
        factor = 0.0  # lambda
        loss += factor * reg_loss
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(network.parameters(), 15)
        optimizer.step()

def test(model):
    model.eval()
    testLoss = 0
    correct = 0
    if cudaFlag:
        model = model.cuda()
    with torch.no_grad():
        for data, target in testLoader:
            if cudaFlag:
                target = target.cuda()
                data = data.cuda()
            output = model(data)
            testLoss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    testLoss /= len(testLoader.dataset)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        testLoss, correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset)))
    return correct / len(testLoader.dataset)