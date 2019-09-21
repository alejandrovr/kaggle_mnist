#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:15:03 2019

@author: alejandro
"""

import torch
import pandas as pd
import numpy as np
import glob

from mnist_newbie.net import SimpleNet
from mnist_newbie.utils import batchatalize


def get_accuracy(pred_idx, real_idx):
    count = 0
    for yh, y in zip(pred_idx,real_idx):
        if yh == y:
            count+=1
            
    return count/len(real_idx)


train_csv = '~/Kaggle/mnist_newbie/digit-recognizer/train.csv'
test_csv = '~/Kaggle/mnist_newbie/digit-recognizer/test.csv'
net = SimpleNet(784, 100, 10) #pixels, hidden cells, output
n_batches = 1000
lr = 1e-1
device = torch.device("cpu")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
df = pd.read_csv(train_csv)

msk = np.random.rand(len(df)) < 0.8
pd_train = df[msk]
pd_test = df[~msk]
print('Training size:',len(pd_train))
print('Test size:',len(pd_test))

train_log = []
test_log = []

net.to(device)

for i in range(n_batches):
    net = net.train()
    print(i,'/',n_batches)
    batch = batchatalize(pd_train, batch_size=100,flat=True)
    train_x = np.array([px_val for px_val, _ in batch])
    train_y = np.array([label for _, label in batch])

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).long()
    
    train_x.to(device)
    train_y.to(device)
    
    yhat = net(train_x)
    loss = criterion(yhat,train_y)
    print('Loss',loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    pred_idx = yhat.argmax(dim=1).detach().cpu().numpy().flatten()
    real_idx = train_y.detach().cpu().numpy().flatten()
    print('Accuracy', get_accuracy(pred_idx, real_idx))
    train_log.append(get_accuracy(pred_idx, real_idx))
    
    if i % 10 == 0:
        net = net.eval()
        print(i,'/',n_batches)
        test_batch = batchatalize(pd_test, batch_size=100,flat=True)
        test_x = np.array([px_val for px_val, _ in test_batch])
        test_y = np.array([label for _, label in test_batch])
    
        test_x = torch.from_numpy(test_x).float()
        test_y = torch.from_numpy(test_y).long()
        
        test_x.to(device)
        test_y.to(device)
        
        yhat = net(test_x)
        loss = criterion(yhat, test_y)
        print('TEST Loss:', loss.item())
        optimizer.zero_grad()
        
        pred_idx = yhat.argmax(dim=1).detach().cpu().numpy().flatten()
        real_idx = test_y.detach().cpu().numpy().flatten()
        test_acc = get_accuracy(pred_idx, real_idx)
        print('\n\nTEST accuracy:', test_acc)
        test_log.append(test_acc)
    
import matplotlib.pyplot as plt
plt.plot(train_log)  
plt.plot(test_log)   
plt.show()
    
model_int = [int(path.split('-')[-1].split('.')[0]) for path in glob.glob('models/fc_net-*.torch')]
next_idx = 0 if len(model_int)==0 else sorted(model_int)[-1] + 1
torch.save(net.state_dict(), 'models/fc_net-{}.torch'.format(next_idx))