#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:15:03 2019

@author: alejandro
"""

import torch
import pandas as pd
import numpy as np

from mnist_newbie.net import SimpleNet
from mnist_newbie.utils import batchatalize

train_csv = '~/Kaggle/mnist_newbie/digit-recognizer/train.csv'
net = SimpleNet(784, 100, 10) #pixels, hidden cells, output
n_batches = 10
lr = 1e-1
device = torch.device("cpu")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
pd_train = pd.read_csv(train_csv)

net.to(device)
net.train()

for i in range(n_batches):
    print(i,'/',n_batches)
    first_batch = batchatalize(pd_train, batch_size=100,flat=True)
    train_x = np.array([px_val for px_val, _ in first_batch])
    train_y = np.array([label for _, label in first_batch])
    
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    
    train_x.to(device)
    train_y.to(device)
    
    yhat = net(train_x)