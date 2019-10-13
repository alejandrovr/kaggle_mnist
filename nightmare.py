#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:55:00 2019

@author: alejandro
"""

import torch
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import random

from mnist_newbie.net import DeeperCNN_VS, SimpleCNN, DeeperCNN, DeeperCNN2, EvenDeeperCNN_VS
from mnist_newbie.utils import batchatalize, row2np


def fuse_batches(nightmare_batch, failing_examples):
    nightmare_batch = nightmare_batch + failing_examples
    nightmare_batch = nightmare_batch[len(failing_examples):]
    return nightmare_batch

def still_patient(log, threshold=0.005, lastN=10):
    if np.std(log[-lastN:]) < threshold and len(log)>lastN and np.mean(log[-lastN:])>0.99:
        return False
    return True

def get_accuracy(pred_idx, real_idx):
    count = 0
    for yh, y in zip(pred_idx,real_idx):
        if yh == y:
            count+=1
            
    return count/len(real_idx)


def make_submission(test_csv,run=0,view=False):
    import statistics
    device = torch.device("cpu")
    all_nets_kfolds = []
    
    for net_weights in glob.glob('models/net_kfold_run_{}_*_*.torch'.format(run)):
        net = EvenDeeperCNN_VS()
        net.load_state_dict(torch.load(net_weights))
        net = net.eval()
        net = net.to(device)
        all_nets_kfolds.append(net)
        
    id_label = []
    df = pd.read_csv(test_csv)
    total_Test = len(df)
    for idx_row, row in df.iterrows():
        print(idx_row,'/',total_Test)
        npimage = row2np(row,flat=False)
        npimage = npimage[np.newaxis, np.newaxis,:]
        test_x = torch.from_numpy(npimage).float()  
        test_x.to(device)
        
        yiii_hats = []
        for net_kfold in all_nets_kfolds:
            yhat = net_kfold(test_x).detach().cpu().numpy()
            yiii_hats.append(yhat)
            
        result = np.array(yiii_hats)[:,0,:]
        pred_label_kfolds = np.argmax(result,axis=1)
        #TODO: take the most confident one?
        try:
            pred_label = statistics.mode(pred_label_kfolds)
        except:
            pred_label = pred_label_kfolds[0] #TODO:use model with highest accu
        if view:
            plt.close()
            print(pred_label,' | ',result)
            plt.imshow(npimage[0][0])
            plt.show()
            input('next?')

        id_label.append((idx_row + 1, pred_label))
        
    submission_df = pd.DataFrame.from_records(id_label)
    submission_df.columns = ['ImageId', 'Label']
    submission_df.to_csv('my_submission.csv',index=False)
    
    return submission_df
  
train_csv = '~/Kaggle/mnist_newbie/digit-recognizer/train.csv'
test_csv = '~/Kaggle/mnist_newbie/digit-recognizer/test.csv'

final_Test_all = []
test_logs = []

df = pd.read_csv(train_csv)
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
splits = kf.get_n_splits(df)
k_fold_index = 0
run = 7
n_epochs = 8
batch_size = 30
for train_index, test_index in kf.split(df): #each run is a k-fold
    
    pd_train = df.iloc[train_index]
    pd_test = df.iloc[test_index]
    print('Training size:',len(pd_train))
    print('Test size:',len(pd_test))

    train_log = []
    test_log = [0.5]
    net = EvenDeeperCNN_VS()
    n_batches = int((len(pd_train) / batch_size) * n_epochs) #2500
    lr = 1e-1
    device = torch.device("cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.9)
    net.to(device)   
    
    for i in range(n_batches):
        net = net.train()
        batch = batchatalize(pd_train, batch_size=batch_size,flat=False, zoom=False)
        
        train_x = np.array([px_val for px_val, _ in batch])
        train_y = np.array([label for _, label in batch])
        try:
            train_x = train_x[:,np.newaxis, :]
        except:
            continue
        train_x = torch.from_numpy(train_x).float()
        train_y = torch.from_numpy(train_y).long()
        
        train_x.to(device)
        train_y.to(device)
        yhat = net(train_x)
        loss = criterion(yhat,train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        pred_idx = yhat.argmax(dim=1).detach().cpu().numpy().flatten()
        real_idx = train_y.detach().cpu().numpy().flatten()
        broken_idx = np.where(pred_idx!=real_idx)[0].flatten()
        failed = [batch[idx] for idx in broken_idx]
        
        #print('Acc:', get_accuracy(pred_idx, real_idx),'|',i,'/',n_batches,'| N=',len(batch),k_fold_index)
        train_log.append(get_accuracy(pred_idx, real_idx))
        
        if i % 25 == 0:
            net = net.eval()
            test_batch = batchatalize(pd_test, batch_size=500, flat=False)
            
            test_x = np.array([px_val for px_val, _ in test_batch])
            test_y = np.array([label for _, label in test_batch])
        
            test_x = test_x[:,np.newaxis, :]
            test_x = torch.from_numpy(test_x).float()
            test_y = torch.from_numpy(test_y).long()
            
            test_x.to(device)
            test_y.to(device)
            
            yhat = net(test_x)
            loss = criterion(yhat, test_y)
            optimizer.zero_grad()
            
            pred_idx = yhat.argmax(dim=1).detach().cpu().numpy().flatten()
            real_idx = test_y.detach().cpu().numpy().flatten()
            test_acc = get_accuracy(pred_idx, real_idx)
            print('\n\nTEST accuracy:', test_acc,' | Best:',max(test_log))
            print(i,'/',n_batches,'|',k_fold_index)
            test_log.append(test_acc)
            if not still_patient(test_log, threshold=0.1, lastN=5):
                break
    
    test_logs.append((train_log,test_log))
    final_acc = int(np.mean([i for i in test_log[-10:]]) * 100)
    torch.save(net.state_dict(), 'models/net_kfold_run_{}_{}_{}.torch'.format(run,final_acc,k_fold_index))
    k_fold_index += 1

    
for train_log, test_log in test_logs:
    plt.plot(test_log)   
    plt.show()

    
    
make_submission('digit-recognizer/test.csv', run=run, view=False)