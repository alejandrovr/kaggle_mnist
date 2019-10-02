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
import matplotlib.pyplot as plt
import random

from mnist_newbie.net import SimpleNet, SimpleCNN, DeeperCNN, DeeperCNN2
from mnist_newbie.utils import batchatalize, row2np

#TODO:
#Transformations
#Masking
#softmax at the end?

def fuse_batches(nightmare_batch, failing_examples, perc_repl=0.2):
    original_size = len(nightmare_batch)
    fail_size = len(failing_examples)
    topN = int(original_size * perc_repl)
    sampleK = topN if topN < fail_size else fail_size
    nightmare_batch = nightmare_batch[topN:]
    nightmare_batch += random.sample(failing_examples,sampleK)
    nightmare_batch = nightmare_batch[:original_size]
    return nightmare_batch

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
net = DeeperCNN2()
n_batches = 2500 #2500
lr = 1e-1
device = torch.device("cpu")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500,2000], gamma=0.1)
df = pd.read_csv(train_csv)

msk = np.random.rand(len(df)) < 0.98
pd_train = df[msk]
pd_test = df[~msk]
print('Training size:',len(pd_train))
print('Test size:',len(pd_test))

train_log = []
test_log = []

#initialize nightmare batch with random sample
nightmare_batch = batchatalize(pd_train, batch_size=100,flat=False)

net.to(device)

for i in range(n_batches):
    net = net.train()
    print(i,'/',n_batches)
    batch = batchatalize(pd_train, batch_size=100, flat=False)
    nightmares = random.sample(nightmare_batch, len(nightmare_batch))
    plt.imshow(nightmares[0][0])
    plt.show()
    batch += random.sample(nightmares, len(nightmares))
    train_x = np.array([px_val for px_val, _ in batch])
    train_y = np.array([label for _, label in batch])

    train_x = train_x[:,np.newaxis, :]
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).long()
    
    train_x.to(device)
    train_y.to(device)
    yhat = net(train_x)
    loss = criterion(yhat,train_y)
    print('Loss',loss.item())
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    pred_idx = yhat.argmax(dim=1).detach().cpu().numpy().flatten()
    real_idx = train_y.detach().cpu().numpy().flatten()
    failing_examples = np.where(pred_idx!=real_idx)[0].tolist()
    failing_examples = [batch[i] for i in failing_examples]
    
    nightmare_batch = fuse_batches(nightmare_batch, failing_examples, perc_repl=0.5)
    print('Accuracy', get_accuracy(pred_idx, real_idx))
    train_log.append(get_accuracy(pred_idx, real_idx))
    
    if i % 10 == 0:
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
    
    
#CHECK WHICH IMAGES ARE MISCLASSIFIED
net = net.eval()
print(i,'/',n_batches)
test_batch = batchatalize(pd_test, batch_size=len(pd_test),flat=False)
test_x = np.array([px_val for px_val, _ in test_batch])
test_y = np.array([label for _, label in test_batch])

test_x = test_x[:,np.newaxis, :]
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).long()

test_x.to(device)
test_y.to(device)

yhat = net(test_x)
pred_idx = yhat.argmax(dim=1).detach().cpu().numpy().flatten()
real_idx = test_y.detach().cpu().numpy().flatten()

counter = 0
misclassified_idx = []
for pi,ri in zip(pred_idx,real_idx):
    if pi != ri:
        misclassified_idx.append((counter,pi,ri))
    counter+=1
        
for failed in misclassified_idx:
    idx, pred, reality = failed
    plt.imshow(test_x[idx][0])
    plt.title('{}_{}'.format(pred,reality))
    plt.show()
    input('Next?')
    plt.close()
    

#%matplotlib auto
plt.plot(train_log)  
plt.plot(test_log)   
plt.show()
    




model_int = [int(path.split('-')[-1].split('.')[0]) for path in glob.glob('models/fc_net-*.torch')]
next_idx = 0 if len(model_int)==0 else sorted(model_int)[-1] + 1
torch.save(net.state_dict(), 'models/fc_net-{}.torch'.format(next_idx))


make_submission(net, 'digit-recognizer/test.csv')