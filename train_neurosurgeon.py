#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 22:09:50 2019

@author: alejandro
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:15:03 2019

@author: alejandro
"""
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import glob
import torch.optim as optim
from torch.optim import lr_scheduler
from mnist_newbie.net import SimpleNet, SimpleCNN, DeeperCNN
from mnist_newbie.utils import batchatalize, row2np

#TODO:
#Transformations
#Masking
#softmax at the end?

def get_accuracy(pred_idx, real_idx):
    count = 0
    for yh, y in zip(pred_idx,real_idx):
        if yh == y:
            count+=1
            
    return count/len(real_idx)


def make_submission(model, test_csv):
    #ImageId,Label
    #1,3
    #... 
    device = torch.device("cpu")
    model = model.eval()
    model.to(device)
    id_label = []

    df = pd.read_csv(test_csv)
    for idx_row, row in df.iterrows():
        npimage = row2np(row,flat=False)
        npimage = npimage[np.newaxis, np.newaxis,:]
        test_x = torch.from_numpy(npimage).float()  
        test_x.to(device)
        yhat = model(test_x)        
        pred_label = yhat.argmax(dim=1).detach().cpu().numpy().flatten().item()
        id_label.append((idx_row + 1, pred_label))
        
    submission_df = pd.DataFrame.from_records(id_label)
    submission_df.columns = ['ImageId', 'Label']
    submission_df.to_csv('my_submission.csv',index=False)
    
    return submission_df
        
        
train_csv = '~/Kaggle/mnist_newbie/digit-recognizer/train.csv'
test_csv = '~/Kaggle/mnist_newbie/digit-recognizer/test.csv'
#net = SimpleNet(784, 500, 10) #pixels, hidden cells, output
#net = SimpleCNN()

net = models.resnet18(pretrained=True)

for param in net.parameters():
    param.requires_grad = False

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)


n_batches = 1000 #2500
device = torch.device("cpu")

criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(net.fc.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.99)
df = pd.read_csv(train_csv)

msk = np.random.rand(len(df)) < 0.95
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
    batch = batchatalize(pd_train, batch_size=10,flat=False,zoom=True)
    train_x = np.array([px_val for px_val, _ in batch])
    train_y = np.array([label for _, label in batch])

    train_x = train_x[:, np.newaxis, :]
    train_x = np.repeat(train_x, 3, axis=1)
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).long()
    
    train_x.to(device)
    train_y.to(device)
    yhat = net(train_x)
    loss = criterion(yhat,train_y)
    print('Loss',loss.item())
    loss.backward()
    optimizer_ft.step()
    exp_lr_scheduler.step()
    optimizer_ft.zero_grad()
    
    pred_idx = yhat.argmax(dim=1).detach().cpu().numpy().flatten()
    real_idx = train_y.detach().cpu().numpy().flatten()
    print('Accuracy', get_accuracy(pred_idx, real_idx))
    train_log.append(get_accuracy(pred_idx, real_idx))
    
    if i % 10 == 0:
        continue
        net = net.eval()
        print(i,'/',n_batches)
        test_batch = batchatalize(pd_test, batch_size=300,flat=False)
        test_x = np.array([px_val for px_val, _ in test_batch])
        test_y = np.array([label for _, label in test_batch])
    
        test_x = test_x[:,np.newaxis, :]
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
    
#%matplotlib auto
import matplotlib.pyplot as plt
plt.plot(train_log)  
plt.plot(test_log)   
plt.show()
    
model_int = [int(path.split('-')[-1].split('.')[0]) for path in glob.glob('models/fc_net-*.torch')]
next_idx = 0 if len(model_int)==0 else sorted(model_int)[-1] + 1
torch.save(net.state_dict(), 'models/fc_net-{}.torch'.format(next_idx))


make_submission(net, 'digit-recognizer/test.csv')