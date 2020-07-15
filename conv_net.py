##here is the implem1entation of the convolutional neural network!
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class convNet(torch.nn.Module):
    def __init__(self, lrate, in_size, out_size):
    
    super(convNet, self).__init__()
    
    self.conv1 = nn.Conv2d(1, 10, 5)
    self.hidden1 = nn.linear(320, 100) #put the pooled features through a hidden layer
    self.output = nn.linear(100, 10) #this layer classifies for us

    self.reLu = nn.ReLU()
    self.pool = nn.MaxPool2d(2,2)

    
    self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=0.9)
    self.loss_fn = nn.CrossEntropyLoss
    


    def get_parameters(self):
        return self.parameters()

    def forward(self, x)
    #implement forward propogation
    
    x = x.view(-1, 1, 28, 28)
    x = self.pool(self.relu(self.conv1(x)))
    x = x.view(-1, 10*12*12)
    
    x = self.relu(self.fc1(x))
    x = self.output(x)
    
    return x

    def step(self, x, y)
    #perform a gradient descent step

    outputs = self.forward(x)

    #get the loss and backpropagate it
    loss = self.loss_fn(outputs, y)
    loss.backward()

    self.optimizer.step()
    L = loss.item()

    return L

    def train(self, train_data, train_labels, dev_set, learning_rate, no_epochs, momentum, log_interval):
        return 0

