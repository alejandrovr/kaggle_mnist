#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:39:28 2019

@author: alejandro
"""

import torch
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import random
from sklearn import datasets
import xgboost as xgb

from mnist_newbie.net import DeeperCNN_VS, EvenDeeperCNN_VS
from mnist_newbie.utils import row2np    
    

if __name__=="__main__":
    train_csv = '~/Kaggle/mnist_newbie/digit-recognizer/train.csv'
    test_csv = '~/Kaggle/mnist_newbie/digit-recognizer/test.csv'
    device = 'cpu'
    df = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df = df.sample(1000)
    
    
    net_kfold = EvenDeeperCNN_VS()
    net_kfold.load_state_dict(torch.load('models/net_kfold_run_6_98_9.torch'))
    outputs= []
    def hook(module, input, output):
        outputs.append(output.detach().cpu().numpy())
    net_kfold.fc1.register_forward_hook(hook)
    net_kfold.to(device)
    
    fake_csv = []
    total_train = len(df)
    for idx_row, row in df.iterrows():
        print(idx_row,'/',total_train)
        row_image = [i for i in row[1:]]
        npimage = row2np(row_image,flat=False)
        npimage = npimage[np.newaxis, np.newaxis,:]
        test_x = torch.from_numpy(npimage).float()  
        test_x.to(device)
        yhat = net_kfold(test_x).detach().cpu().numpy()
        fp = outputs[-1]
        fp = fp[0].tolist()
        outputs = []
        fake_csv.append([fp, row.label])
        

X = np.array([i[0] for i in fake_csv])
y = np.array([i[1] for i in fake_csv])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)   

param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 10} 

steps = 20  # The number of training iterations

print('Lets train...')
model = xgb.train(param, D_train, steps)
print('Done')

from sklearn.metrics import precision_score, recall_score, accuracy_score

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(Y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))



     
        