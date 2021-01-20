from Config import *
from networkStruct import *
import utils
from traintestfunc import *
utils.setSeed(13)


if __name__ == '__main__':
    print('Dataset is ', dataset)
    print('-' * 100)
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum, weight_decay=weight_decay)
    test(model)
    for epoch in range(1, nEpochs + 1):
        train(model,optimizer)
        test(model)


