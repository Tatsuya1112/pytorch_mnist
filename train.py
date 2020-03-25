import os
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn as nn
from tqdm import tqdm
from utils import AverageMeter
import models

import torch.optim as optim

arch_names = list(models.__dict__.keys())

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='TwoLayerNet')
# parser.add_argument('--model', default='SimpleConvNet')
parser.add_argument('--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--epochs', default=20)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--save_path', default=None)
args = parser.parse_args()

args.save_path = './'+args.model+'.pth'

print('Config -----')
for arg in vars(args):
    print('{:}: {:}'.format(arg, getattr(args, arg)))
print('------------')

if not os.path.exists('./data'):
    os.makedirs('./data')

trainset = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

net = models.__dict__[args.model]()
net.to(args.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(args.epochs):
    # train
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    for i,data in enumerate(tqdm(trainloader)):
        inputs, labels = data[0].to(args.device), data[1].to(args.device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), args.batch_size)
        train_accuracy.update(torch.sum(torch.max(outputs, dim=1)[1]==labels).type(torch.float)/labels.shape[0], args.batch_size)

    # test
    test_loss = AverageMeter()
    test_accuracy = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss.update(loss.item(), args.batch_size)
            test_accuracy.update(torch.sum(torch.max(outputs, dim=1)[1]==labels).type(torch.float)/labels.shape[0], args.batch_size)

    print('epoch: {:3d}, train_loss: {:.3f}, train_accuracy: {:.3f}, test_loss: {:.3f}, test_accuracy: {:.3f}'.format(epoch, train_loss.avg, train_accuracy.avg, test_loss.avg, test_accuracy.avg))

torch.save(net.state_dict(), args.save_path)
print('finished!')