import numpy as np
from matplotlib import pyplot as plt
import copy
import pandas as pd

#%matplotlib inline

from IPython.display import Image, display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image

import docopt


def str_to_int_list(string):
    return [int(s) for s in string.split(',')]


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
    #print(train_dataset[0][0])

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
        elif func=='celu':
            self.layers.append(nn.CELU())
        elif func=='tanhshrink':
            self.layers.append(nn.Tanhshrink())

    def construct_net(self, layer_sizes, fn, device):
        self.layers = []
        # Add first input layer
        self.layers.append(nn.Linear(28*28, layer_sizes[0]))
        for idx, layer in enumerate(layer_sizes):
            if idx==len(layer_sizes) - 1:
                # We're at the last layer, so break out of loop and add
                # output layer
                break
            if not layer==min(layer_sizes):
                self.add_func(fn)
            self.layers.append(nn.Linear(layer_sizes[idx],
                layer_sizes[idx + 1]))

        self.add_func(fn)
        # Add output layer
        self.layers.append(nn.Linear(layer_sizes[-1], 28*28))
        #self.add_func('tanhshrink')
        self.net = nn.Sequential(*self.layers)
        for layer in self.net:
            layer.to(device)
        print(self.net)

    def forward(self, x):
        return self.net(x)


def train(model,
        device,
        train_loader,
        optimizer,
        criterion,
        epoch,
        log_interval=50,
        compare=None):
    model.train()
    #correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28) # flatten into 1D array for dense nn
        #print(max(data[0]))
        #print(min(data[0]))
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
            #print(max(output[0]))
            if compare:
                compare_obj = getattr(nn, compare)()
                compare_loss = compare_obj(output, target)
                print('Loss for {} (not trained for this loss fn):'.format(compare))
                print('{}'.format(compare_loss))

    print('Train Epoch: {}, Loss: {:.6f}'.format(
        epoch, loss.item()))
    return loss.item()
    #print(target)
    #print(output)


def evaluate(model,
        device,
        test_loader,
        criterion):
    losslist = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.view(-1, 28*28)
            target = data.to(device)
            output = model(target)
            loss = criterion(output, target)
            losslist.append(loss.item())

    return sum(losslist) / len(losslist)


def plot_reconstructions(model, train_loader, device,
        out='training_reconstruction.png'):
    # Extract 6 figures from training DataLoader
    mini_batch, _ = next(iter(train_loader))
    n_examples = min(6, mini_batch.shape[0])
    examples = mini_batch[:n_examples]

    # Compute reconstructions
    with torch.no_grad():
        reconstr_examples = model.forward(
                examples.view(n_examples, -1).to(device)
        )

    # Save image with original v. reconstructed images
    comparison = torch.cat([
        examples,
        reconstr_examples.view(-1, 1, 28, 28).cpu()
        ])
    save_image(comparison.cpu(), out,
            nrow=n_examples)


def imshow(inp, 
           figsize=(10,10),
           mean=0.1307, # for MNIST train
           std=0.3081, # for MNIST train
           title=None,
           onpick=False,
           outfile=None):
    """Imshow for Tensor."""
    inp = inp.cpu().detach()
    if not onpick:
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = inp.numpy()
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=figsize)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    #plt.show()
    if outfile is not None:
        plt.savefig(outfile)


def reconstructions_from_batch(model, batch, device):
    batch = batch.view(-1, 28 * 28).to(device)
    reconstruction = model(batch)
    return reconstruction.reshape(batch.shape[0],1,28,28)


def loss_curve(df, out='loss_curve.png',
        figsize=(10,10)):
    plt.figure(figsize=figsize)
    x = df['epoch']
    y1 = df['loss']
    y2 = df['test']
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.plot(x, y1, label='Avg. loss over training set', linestyle='-', marker='o')
    plt.plot(x, y2, label='Avg. loss over test set', linestyle='-', marker='o')
    plt.legend()
    plt.show()


def encode(model, device, test_loader):
    encoder = model.net[:len(model.net)//2]
    print('ENCODER:')
    print(encoder)
    num_elements = len(test_loader.dataset)
    print('Running encoder on {} elements'.format(num_elements))
    num_batches = len(test_loader)
    print('Number of batches: {}'.format(num_batches))
    batch_size = test_loader.batch_size
    predictions = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if batch_idx == num_batches - 1:
                end = num_elements
            data = data.view(-1, 28*28)
            data = data.to(device)
            output = encoder(data)
            predictions.append(output)
            labels.append(label)
    
    predictions = torch.cat(predictions)
    predictions.to(device)
    labels = torch.cat(labels)
    labels.to(device)

    return predictions, labels


def decode(model, device, coords):
    # A better way to do this would be to use argmin, but this works for
    # now as long as I always use the same number of layers on each
    # side.
    decoder = model.net[len(model.net)//2:]
    print('DECODER:')
    print(decoder)

    with torch.no_grad():
        data = coords
        data = data.to(device)
        output = decoder(data)
    output = output.view(28,28)
    return output
    


def plot_latentspace(predictions, labels, model, device):
    predictions = predictions.cpu()
    labels = labels.cpu()
    df = pd.DataFrame(predictions.numpy())
    df['labels'] = labels.numpy()

    def on_pick(event):
        ix, iy = event.xdata, event.ydata
        print('Picked coordinates ({}, {})'.format(ix, iy))

        coords = torch.FloatTensor([ix, iy])
        output = decode(model, device, coords)
        imshow(output, onpick=True)

    colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red',
            4:'tab:purple', 5:'tab:brown', 6:'tab:pink', 7:'tab:gray',
            8:'tab:olive', 9:'tab:cyan'}
    print(df)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for key, group in df.groupby('labels'):
        ax.scatter(group[0], group[1], c=colors[key],
                label=key)
    ax.legend()
    plt.xlabel('Latent variable 1')
    plt.ylabel('Latent variable 2')
    fig.canvas.mpl_connect('button_press_event', on_pick)

    plt.show()


