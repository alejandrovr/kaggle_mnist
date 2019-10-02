#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 00:31:16 2019

@author: alejandro
"""

import torch
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import random

from mnist_newbie.net import DeeperCNN_VS, SimpleCNN, DeeperCNN, DeeperCNN2
from mnist_newbie.utils import batchatalize, row2np


#TODO:
# fussion two images 80 vs 20%, label as 80
# zoom, rotations and displacements

def get_accuracy(pred_idx, real_idx):
    count = 0
    for yh, y in zip(pred_idx,real_idx):
        if yh == y:
            count+=1
            
    return count/len(real_idx)


def make_submission(test_csv,view=False):
    device = torch.device("cpu")
    net0 = DeeperCNN_VS()
    net0.load_state_dict(torch.load('models/net0_vsALL.torch'))
    net0.eval()
    
    net1 = DeeperCNN_VS()
    net1.load_state_dict(torch.load('models/net1_vsALL.torch'))
    net1.eval()

    net2 = DeeperCNN_VS()
    net2.load_state_dict(torch.load('models/net2_vsALL.torch'))
    net2.eval()

    net3 = DeeperCNN_VS()
    net3.load_state_dict(torch.load('models/net3_vsALL.torch'))
    net3.eval()

    net4 = DeeperCNN_VS()
    net4.load_state_dict(torch.load('models/net4_vsALL.torch'))
    net4.eval()

    net5 = DeeperCNN_VS()
    net5.load_state_dict(torch.load('models/net5_vsALL.torch'))
    net5.eval()

    net6 = DeeperCNN_VS()
    net6.load_state_dict(torch.load('models/net6_vsALL.torch'))
    net6.eval()

    net7 = DeeperCNN_VS()
    net7.load_state_dict(torch.load('models/net7_vsALL.torch'))
    net7.eval()

    net8 = DeeperCNN_VS()
    net8.load_state_dict(torch.load('models/net8_vsALL.torch'))
    net8.eval() 
    
    net9 = DeeperCNN_VS()
    net9.load_state_dict(torch.load('models/net9_vsALL.torch'))
    net9.eval()     
    
    net0.to(device)
    net1.to(device)
    net2.to(device)
    net3.to(device)
    net4.to(device)
    net5.to(device)
    net6.to(device)
    net7.to(device)
    net8.to(device)
    net9.to(device)
    id_label = []

    df = pd.read_csv(test_csv)
    total_Test = len(df)
    for idx_row, row in df.iterrows():
        print(idx_row,'/',total_Test)
        npimage = row2np(row,flat=False)
        npimage = npimage[np.newaxis, np.newaxis,:]
        test_x = torch.from_numpy(npimage).float()  
        test_x.to(device)
        yhat0 = net0(test_x).detach().cpu().numpy()
        yhat1 = net1(test_x).detach().cpu().numpy()
        yhat2 = net2(test_x).detach().cpu().numpy()
        yhat3 = net3(test_x).detach().cpu().numpy()
        yhat4 = net4(test_x).detach().cpu().numpy()
        yhat5 = net5(test_x).detach().cpu().numpy()
        yhat6 = net6(test_x).detach().cpu().numpy()
        yhat7 = net7(test_x).detach().cpu().numpy()
        yhat8 = net8(test_x).detach().cpu().numpy()
        yhat9 = net9(test_x).detach().cpu().numpy()
        
        result = [yhat0, yhat1, yhat2, yhat3, yhat4, yhat5, yhat6, yhat7, yhat8, yhat9]
        result = np.array(result)[:,0,:]
        pred_label = np.argmax(result[:,1])
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

df = pd.read_csv(train_csv)
msk = np.random.rand(len(df)) < 0.95
pd_train = df[msk]
pd_test = df[~msk]
print('Training size:',len(pd_train))
print('Test size:',len(pd_test))

final_Test_all = []
test_logs = []
for chosen_digit in range(10):
    train_log = []
    test_log = []
    net = DeeperCNN_VS()
    n_batches = 5500 #2500
    lr = 1e-1
    device = torch.device("cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500,2000], gamma=0.5)
    net.to(device)
    
    for i in range(n_batches):
        net = net.train()
        batch = batchatalize(pd_train, batch_size=50, flat=False)
        batch1 = [[i[0], 1]  for i in batch if i[1]==chosen_digit]
        batch1_notone = [[i[0], 0] for i in batch if i[1]!=chosen_digit][:len(batch1)]
        batch = batch1 + batch1_notone
        
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
        print('Acc:', get_accuracy(pred_idx, real_idx),'|',i,'/',n_batches)
        train_log.append(get_accuracy(pred_idx, real_idx))
        
        if i % 10 == 0:
            net = net.eval()
            test_batch = batchatalize(pd_test, batch_size=600,flat=False)
            batch1 = [[i[0], 1]  for i in test_batch if i[1]==chosen_digit]
            batch1_notone = [[i[0], 0] for i in test_batch if i[1]!=chosen_digit][:len(batch1)]
            test_batch = batch1 + batch1_notone        
            
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
            print('\n\nTEST accuracy:', test_acc)
            test_log.append(test_acc)
        
        
    #CHECK WHICH IMAGES ARE MISCLASSIFIED
    net = net.eval()
    print(i,'/',n_batches)
    test_batch = batchatalize(pd_test, batch_size=len(pd_test),flat=False)
    batch1 = [[i[0], 1]  for i in test_batch if i[1]==chosen_digit]
    batch1_notone = [[i[0], 0] for i in test_batch if i[1]!=chosen_digit][:len(batch1)]
    test_batch = batch1 + batch1_notone      
    
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
            
    print('Predicted {} were wrong. Total tested: {}'.format(misclassified_idx,counter))
    final_Test_all.append((chosen_digit,misclassified_idx,counter))    
    
    #%matplotlib auto
    #plt.plot(train_log)  
    plt.plot(test_log)   
    plt.show()
    test_logs.append((train_log,test_log))
    torch.save(net.state_dict(), 'models/net{}_vsALL.torch'.format(chosen_digit))
    

for _, test_log in test_logs:
    plt.plot(test_log)   
    plt.show()
    plt.close()
    
    
    
make_submission('digit-recognizer/test.csv', view=False)