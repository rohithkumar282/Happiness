#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import math
import csv
import time
import matplotlib.pyplot as plt

def initalize_weights(size_layer, size_next_layer):
    np.random.seed(1)
    epsilon = np.sqrt(2.0 / (size_layer * size_next_layer) )
    w = epsilon * (np.random.randn(size_next_layer, size_layer))
    return w.transpose()

def normalize_data(data):
    data_new = data.copy()
    max_num = data.max()
    min_num = data.min()
    data_new = (data - min_num)/(max_num - min_num)
    return data_new

def load_file():
    hp1=pd.read_csv('2017.csv');
    hp2=pd.read_csv('2016.csv');
    hp3=pd.read_csv('2015.csv'); 
    col_2015=['Region','Standard Error'];
    hp3_mod=hp3.drop(col_2015,axis=1);
    col_2016=['Region','Lower Confidence Interval','Upper Confidence Interval'];
    hp2_mod=hp2.drop(col_2016,axis=1);
    col_2017=['Whisker.high','Whisker.low'];
    hp1_mod=hp1.drop(col_2017,axis=1);
    final=[hp3_mod,hp2_mod,hp3_mod];
    finaltable=pd.concat(final);
    Y = finaltable.iloc[:,2].values;
    X = finaltable.iloc[:,3:].values;
    return X,Y
        
epochs = 1000
loss = np.zeros([epochs,1])

X, Y= load_file()
tic = time.time()
Y=Y.reshape(Y.shape[0],1)
n_examples = X.shape[0]
w1 = initalize_weights(X.shape[1]+1, X.shape[1]+1)
w2 = initalize_weights(X.shape[1]+1, Y.shape[1])
X1=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
for ix in range(epochs):
    n_examples = X1.shape[0]   
    a1 = X1
    z2 = a1.dot(w1.T)
    a2 = 1/(1+np.exp(-z2))
    z3 = a2.dot(w2)
    a3 = 1/(1+np.exp(-z3))
    Y_hat = a3
    Y_hat1=normalize_data(Y_hat)
    Y1=normalize_data(Y)
    loss[ix] = (0.5) * np.square(Y_hat1 - Y1).mean()
    d3 = Y_hat1 - Y1
    grad2 = a2.T.dot(d3)/n_examples
    d2 = d3.dot(w2.T)
    d2 = d2*a2*(1-a2) 
    grad1 = a1.T.dot(d2)/n_examples
    w1 = w1 - grad1
    w2 = w2 - grad2
plt.figure()
ix = np.arange(epochs)
plt.plot(ix, loss) 
plt.xlabel('number of epochs')
plt.ylabel('Root mean squared error')
print(loss.min())


# In[ ]:




