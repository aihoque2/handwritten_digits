##here is the implem1entation of the convolutional neural network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class convNet(torch.nn.Module):
    def __init__(self, lrate, in_size, out_size, momentum):
        super(convNet, self).__init__()

        #you need the layers, the loss function, and the optimizer
    
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.hidden1 = nn.linear(320, 100) #put the pooled features through a hidden layer
        self.output = nn.linear(100, out_size) #this layer classifies for us

        self.reLu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)

    
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=momentum)
        self.loss_fn = nn.CrossEntropyLoss
    


    def get_parameters(self):
        return self.parameters()

    def forward(self, x):
        #implement forward propogation
        x = x.view(-1, 1, 28, 28)
        x = self.pool(self.reLu(self.conv1(x)))
        x = x.view(-1, 10*12*12)
        x = self.reLu(self.hidden1(x))
        x = self.output(x)
        return x

    def step(self, x, y):
        #perform a gradient descent step

        outputs = self.forward(x)

        #get the loss and backpropagate it
        loss = self.loss_fn(outputs, y)
        loss.backward()

        self.optimizer.step()
        L = loss.item()
        return L

    def train(self, train_data, train_labels, dev_set, learning_rate, no_epochs, momentum, log_interval):
        
        #TODO: implement training algorithm here
        #return model, losses, yhats

        batch_size=100
        no_iter = int(len(train_data[0])/batch_size)

        #Normalize the data
        mean = train_set.mean(dim=0, keepdim=True)
        stds = train_set.std(dim=0, keepdim=True)
        train_set = (train_set-mean)/stds
        dev_set = (dev_set-mean)/stds

        losses = []
        yhats = []

        ##DECLARE THE MODEL
        model = convNet(lrate=learning_rate, in_size=784, out_size=10, momentum=momentum)



        for i in range(no_iter):
            start_idx = i * batch_size

            if (start_idx >= len(train_set)):
                start_idx = start_idx - len(train_set) #get the proper index

            x = train_set.narrow(0, start_idx, batch_size)
            y = train_set.narrow(0, start_idx, batch_size)
            curr_loss = model.step(x, y)



        outputs=model.forward(dev_set)
        _, predicted = torch.max(outputs, i)

        yhats= np.array(predicted)
        print("number of parameters: ", sum(p.numel() for p in model.parameters())
        return losses, yhats, model     

