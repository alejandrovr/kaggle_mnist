#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:10:31 2019

@author: alejandro
"""

import torch
import torch.nn.functional as F

class SimpleNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SimpleNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
    
class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(1, 18, kernel_size=4, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * 24 * 24, 64)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        #print('0',x.size())
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        #print('1',x.size())
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 24 * 24)
        #print('2',x.size())        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        #print('3',x.size())        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        #print('4',x.size())        
        return(x)
        
        
class DeeperCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(DeeperCNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(96 * 10 * 10, 64)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        #print('0',x.size())
        x = self.pool2(F.relu(self.conv2(x)))
        #print('1',x.size())
        #print('1.5',x.size())
        x = x.view(-1, 96 * 10 * 10)
        #print('2',x.size())        
        x = F.relu(self.fc1(x))
        #print('3',x.size())        
        x = self.fc2(x)
        #print('4',x.size())        
        return(x)
        
        
if __name__ == '__main__':
    import numpy as np
    fake_input = np.random.rand(5,1,28,28)
    fake_input = torch.from_numpy(fake_input).float()
    net = DeeperCNN()
    output = net(fake_input)
    