##here is the implementation of the convolutional neural network!
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class convNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
    
    super(convNet, self).__init__()
    
    self.loss_fn =loss_fn



    def get_parameters(self):
        return self.parameters()

    def forward(self, x)
    #implement the forward algo here!


    def step(self, x, y)
    #perform a gradient descent step







