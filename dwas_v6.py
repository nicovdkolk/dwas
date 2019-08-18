# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:01:14 2019
@author: Brian, Python 3.5

Uses dwas_data.txt with columns: 
Hull,Froude Number,Leeway angle,Heel angle,Fx,std Fx,Fy, std Fy,Mz,std Mz data
The hull number is stored in hulls.txt with columns: 
ID	Description	Length	Beam	Draft	Cp	Cb	Cm	L_B	B_T	T_L	L_vol3	Cwp	AwpSw	Rb_T	Deadrise

The Hull and ID columns are joined so all hull data is added for each force observations

The dependent variables are Fx, Fy, and Mz

They are extracted and new matrices are made before concatenated again so they are at the end

Data is first max-min to 0-1 then shuffled and then split up

Uses an MLP Feed Forward model with 3 outputs

"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Nadam

def minmax(a):
    scaler = MinMaxScaler()
    scaler.fit(a)
    return scaler.transform(a)

# fix random seed for reproducibility
np.random.seed(7)
Xbatch = []
Xlookback= []
Xdropout= []

# get input data
hull_file = 'hulls.txt'
dwas_file = 'dwas_data.txt'
current_path = os.getcwd()  # get current working path to save later

#upload hulls
hulls_df = pd.read_csv(hull_file, engine = 'python', sep= '\s+|\t+|\s+\t+|\t+\s+')
dwas_df = pd.read_csv(dwas_file)

#add hull info to each observation
df_tot = pd.merge(right=dwas_df,left=hulls_df, 
                  how='left', 
                  left_on='ID', 
                  right_on='Hull').dropna(axis=1)  #and drop na columns

df_tot =df_tot.drop(['Description', 'Hull'], axis=1) # delete columns

df_tot = df_tot.rename(columns={"ID": "Hull"}) # change ID column to Hull

df_y = df_tot[['Fx', 'Fy', 'Mz']]
df_x = df_tot.drop(['Hull', 'Fx', 'Fy', 'Mz'], axis=1) 

x_raw = df_x.values
y_raw = np.asmatrix((df_y.values))

x_raw_ct = len(x_raw.T)
y_raw_ct = len(y_raw.T)

tot_par = np.concatenate((x_raw, y_raw), axis = 1) # remerge in order to shuffle

#******************************************************************************

# prepare output training set using read_force function

force_out = minmax(tot_par)
len_force = len(force_out.T)

# make complete matrix to shuffle rows
tot_train = force_out
len_tot = len(tot_train.T)

len_dKE = len(x_raw)

# ask if data should be shuffled
right_vars = 'n'
while right_vars == 'n':
    
    shuffle = False
    
    try:
        n_shuffle = input('Shuffle data? y/n (default = n) ')
    except:
        n_shuffle = 'n'
    
    if n_shuffle == 'n':
        print('No shuffling')
        shuffle = False
    
    else:    
        print('Data will be shuffled')
        shuffle = True
    
    # shuffle/randomize 
    if shuffle == True:
        tot_train = np.take(tot_train, 
                            np.random.permutation(tot_train.shape[0]), 
                                                         axis=0, out= tot_train)
    try:
        right_vars = input('Continue with modeling (y/n)? (Default = y): ')
    except:
        right_vars = 'y'
        
# extract input and output training sets
train_in = tot_train[:,0:x_raw_ct]
train_out = tot_train[:,x_raw_ct:((y_raw_ct+x_raw_ct))]

#get parameters for MLP nodes
tot_set, in_shape = np.shape(train_in)
tot_set, out_shape = np.shape(train_out)

# set up training and testing values
ntrain = int(0.7* tot_set)

x_train = train_in[0:ntrain,:]
y_train = train_out[0:ntrain,:]

x_test = train_in[ntrain:tot_set,:]
y_test = train_out[ntrain:tot_set,:]

#**********************************************************
# build MLP model
#**********************************************************

model = Sequential()

n = in_shape + 10
# add input layer with n output nodes
model.add(Dense(n, input_dim= in_shape, 
                use_bias=True, kernel_initializer='random_uniform', 
                bias_initializer='zeros',
                activation = 'sigmoid'))
model.add(Dropout(0.2))

# add another hidden layer
model.add(Dense(n, activation = 'sigmoid'))
model.add(Dropout(0.2))


# add another hidden layer
model.add(Dense(n, activation = 'sigmoid'))
model.add(Dropout(0.2))

# add output layer
model.add(Dense(out_shape, activation='tanh'))

optmz = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(loss='mean_squared_error',
              optimizer= nadam,
              metrics=['mae','accuracy'])

# print model statistics
model.summary()

batch = 100 # batch size

history = model.fit(train_in, train_out,
          epochs = 100,
          shuffle = True,
          verbose = 1,
          batch_size = batch,
          validation_data =(x_test, y_test))

score = model.evaluate(x_test, y_test, batch_size = batch)

m = 3 # of layers

#plot results
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy ('+str(n)+' nodes, '+str(m)+' layers)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss ('+str(n)+' nodes, '+str(m)+' layers)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

if 1:
    df = pd.DataFrame(history.history)
    writer = pd.ExcelWriter('dwas_mlp_output.xlsx')
    df.to_excel(writer,'DWAS MLP Results')
    writer.save()
    


