#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:15:44 2019

@author: alejandro
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def row2np(row):
    np_pic = np.array(row).reshape(28,28)
    return np_pic

def batchatalize(pd_dataframe,batch_size=100):
    batch = []
    batch_rows = pd_dataframe.sample(batch_size)
    for idx, row in batch_rows.iterrows():
        int_label, pixels = row.label, [i for i in row[1:]]
        np_pic = row2np(pixels)
        batch.append([np_pic,int_label]) 
    return batch
    
if __name__ == "__main__":
    train_csv = '~/Kaggle/mnist_newbie/digit-recognizer/train.csv'
    pd_train = pd.read_csv(train_csv)
    first_batch = batchatalize(pd_train, batch_size=5)
    
    
    for np_pic, label in first_batch:
        plt.imshow(np_pic)
        plt.show()
        print('Image on top is a:',label)
        input('next?')