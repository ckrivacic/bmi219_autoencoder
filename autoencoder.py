import numpy as np
from matplotlib import pyplot as plt

#%matplotlib inline

from IPython.display import Image, display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image


def dataset():
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST(
            './bmi219_downloads', train=True, download=True,
            transform=preprocessing)
    #print(len(train_dataset[0][0][0][0]))
    #print(len(train_dataset[0][0][0]))

    test_dataset = datasets.MNIST(
            './bmi219_downloads', train=False, download=True,
            transform=preprocessing)
    #print(test_dataset)

    return train_dataset, test_dataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim=28*28):
        super(Autoencoder, self).__init__()
        '''
        self.layers = []
        self.layers.append(nn.Linear(28*28, 1000))
        self.layers.append(nn.Linear(1000, 500))
        self.layers.append(nn.Linear(500, 250))
        self.layers.append(nn.Linear(250, 2))
        self.layers.append(nn.Linear(2, 250))
        self.layers.append(nn.Linear(250, 500))
        self.layers.append(nn.Linear(500, 1000))

        self.net = nn.Sequential(*layers)
        '''

    def add_func(self, func):
        if func=='relu':
            self.layers.append(nn.ReLU())
        elif func=='softmax':
            self.layers.append(nn.Softmax())
        elif func=='tanh':
            self.layers.append(nn.Tanh())
        elif func=='leakyrelu':
            self.layers.append(nn.LeakyReLU())
        elif func=='logsigmoid':
            self.layers.append(nn.LogSigmoid())
        elif func=='prelu':
            self.layers.append(nn.PReLU())
        elif func=='sigmoid':
            self.layers.append(nn.Sigmoid())

    def construct_net(self, layer_sizes, fn):
        self.layers = []
        self.layers.append(nn.Linear(28*28, layer_sizes[0]))
        for idx, layer in enumerate(layer_sizes):
            #if idx==0:
            #    continue
            if idx==len(layer_sizes) - 1:
                break
            self.add_func(fn)
            self.layers.append(nn.Linear(layer_sizes[idx],
                layer_sizes[idx + 1]))
            #self.add_func(fn)
        self.add_func(fn)
        self.layers.append(nn.Linear(layer_sizes[-1], 28*28))
        self.add_func('tanh')
        self.net = nn.Sequential(*self.layers)
        print(self.net)

    def forward(self, x):
        return self.net(x)


def train(model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=128):
    model.train()
    #correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28) # flatten into 1D array for dense nn
        target = data.to(device)#, target.to(device)
        optimizer.zero_grad() # reset gradient each epoch
        output = model(target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #pred = output.argmax(dim=1, keepdim=True) # Get index of max log-probability
        #correct += pred.eq(target.view_as(pred)).sum().item() # tally # correct

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('Train Epoch: {}, Loss: {:.6f}'.format(
        epoch, loss.item()))
    print(target)
    print(output)

if __name__=='__main__':
    BATCH_SIZE=128
    NUM_WORKERS=4
    use_cuda=False
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(7)

    train_dataset, test_dataset = dataset()

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS)

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS)
    '''
    for tup in iter(train_loader):
        tup[1] = tup[0]
    print(next(iter(train_loader)))
    '''

    net = Autoencoder()
    if use_cuda:
        net.cuda()
    net.construct_net([1000, 500, 250, 2, 250, 500, 1000], 'relu')
    #net.construct_net([1000,2,100], 'sigmoid')

    optimizer = optim.Adam(net.parameters(),
            lr = 0.001)
    criterion = nn.MSELoss()

    epochs = 10
    for epoch in range(epochs):
        train(net, device,
                train_loader,
                optimizer,
                criterion,
                epoch)

'''
> Question 0.1) Why is it important to set the seed for the random number
generator?

This is probably used to seed the weights. If we want to play around
with hyperparameters, it's best to compare networks that have been
seeded with the same rng. (?)

> **Q1.1) How many examples do the training set and test set have?**

Training: 60000
Test: 10000

> **Q1.2) What's the format of each input example? Can we directly put these into a fully-connected layer?**

Each input is a tuple of a 28x28 matrix of pixels and a label (0-9).


> **Q1.3) Why do we normalize the input data for neural networks?**

Large numbers can make the gradients very shallow during
back-propogation, making training very slow.


> **Q1.4) In this scenario, MNIST is already split into a training set 
and a test set. What is the purpose of dataset splitting (and specifically, 
the purpose of a test set)? For modern deep learning, a three-way split 
into training, validation, and test sets is usually preferred, why?**

It is possible for a neural net to memorize many of the features of the
training set. The test set ensures that the neural net works on data
that it hasn't ever seen before, ensuring generalizability.
While a test set is used to evaluate the model parameters, a validation
set is used to evaluate the model hyperparameters. The reason it is
preferred to have a separate validation and test set is that the
hyperpararmeters may be biased towards the validation set.


> **Q2.1) It's recommended to shuffle the training data over each epoch, 
but this isn't typically the case for the test set, why?**

Shuffling during training introduces stochasticity to the training
process, which is necessary because we are not guaranteed to find the
global minimum. This is unnecessary during the testing phase because we
are not changing the model parameters, just evaluating them.


> **Q2.2) What seems to be a good batch size for training? What happens if you train 
with a batch size of 1? What about a batch size equal to the total training set?**

A batch size of 1 trains much slower because it means you can't
parallelize the training. Training on the whole dataset means you can
only learn about the averages of features. A good batch size for
training seems to be somewhere around train_size / ~400-500.

> **Q2.3) The PyTorch DataLoader object is an iterator that generates batches as it's called. 
Try to pull a few images from the training set to see what these images look like. 
Does the DataLoader return only the images? What about the labels?**

It returns both the image (in 28x28 matrix/tensor form) and the label.
The labels are typically required for evaluating the model (though this
is not the case in an autoencoder).


> **Q3.1) What activation functions did you use, and why?**
Starting with ReLU because if the initial weights are way off, it has a
relatively strong gradient it can follow.
'''
