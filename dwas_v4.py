# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:49:55 2019

@author: Brian
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Nadam

def onehot(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def minmax(a):
    scaler = MinMaxScaler()
    scaler.fit(a)
    return scaler.transform(a)
    
#read datafile

xlsfile="DWAS.xlsx"
df = pd.read_excel(xlsfile,sheet_name="data_raw")
geo= pd.read_excel(xlsfile,sheet_name="geo")

dwas=pd.merge(geo,df,how='outer',on='ID')

#convert hull ID column to one-hot encoding
ID_onehot = onehot(np.asarray(dwas.ID))

#scale continuous data (0-1)
a = (dwas.loc[:, dwas.columns != 'ID']).values
a_scaled = minmax(a)
a_random= shuffle(a_scaled, random_state=0)

#separate input features from output features

ind_in=np.arange(1,14)

BIGscore=np.empty([np.size(ind_in,0)-2,3])

#jj is dropped variable (speed/leeway not included)
for jj in range(0,np.size(ind_in,0)-1 ):

    if jj == 0:
        #complete set as reference
        in_f = a_random[:,ind_in]
    else:
        #remove one variable
        in_f = a_random[:,np.delete(ind_in,jj-1)]
    
    out_f = a_random[:,13:16]
    
    
    train_in = in_f
    tot_set, in_shape = np.shape(train_in)
    
    ntrain = int(0.7*tot_set)
    
    x_train = train_in[0:ntrain,:]
    y_train = out_f[0:ntrain,:]
    
    x_test = train_in[ntrain:tot_set,:]
    y_test = out_f[ntrain:tot_set,:]
    
    
    #build MLP model
    model = Sequential()
    
    #n nodes
    n = 100
    
    #m hidden layers
    m=2
    
    for ii in range(0,m):
    
        model.add(Dense(n, input_dim= in_shape, activation = 'tanh'))
        model.add(Dropout(0.2))
    
    #add output layer
    model.add(Dense(3, activation='sigmoid'))
    
    optmz = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='mean_absolute_error',
                  optimizer=optmz,
                  metrics=['mae','accuracy'])
    
    model.summary()
    
    history = model.fit(x_train, y_train,
              epochs=40,
              shuffle=True,
              verbose = 0,
              batch_size=10,
              validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, batch_size=10)
    
    ##plot results
    if 0:
        fig, ax1 = plt.subplots()
        
        ax1.set_xlabel('Epoch')
        ax1.plot(history.history['acc'])
        ax1.plot(history.history['val_acc'])
        ax1.set_ylabel('Accuracy')
        ax1.legend(['Train', 'Test'], loc='right')
        ax1.set_ylim([.6,1])
        
        ax2=ax1.twinx()
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_ylabel('Loss')
        ax2.set_ylim([0,.2])
        
        plt.title('Model Accuracy / Loss : ('+str(n)+' nodes, '+str(m)+' layers)')
        fig.tight_layout()
        plt.show()
    if jj==0:
        ref=score
    else:
        BIGscore[jj-1]=list(np.array(ref) - np.array(score))

fig, ax1 = plt.subplots()
ax1.set_xticks(np.arange(1,13))
ax1.set_xticklabels(list(dwas.iloc[:,ind_in]))
ax1.set_xlabel('Dropped input')
ax1.set_ylabel('rel. Score')

ax1.plot(ind_in[0:-2],BIGscore[:,0] )
ax1.plot(ind_in[0:-2],BIGscore[:,2])

plt.title('Change in Model Accuracy / Loss : ('+str(n)+' nodes, '+str(m)+' layers)')
ax1.legend(['Loss','Acc'])
fig.tight_layout()
plt.show()
if 1:
    df = pd.DataFrame(history.history)
    writer = pd.ExcelWriter('dwas_mlp_output.xlsx')
    df.to_excel(writer,'DWAS MLP Results')
    writer.save()
