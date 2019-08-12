# -*- coding: utf-8 -*-
"""
Updated FFNN to accept component kinetic energy input to generate
force, pressure and sheer components
Fx, Fy, Sx, Sy, Px, Py files must be in same folder as program and dKE file
Removed first and last columns to avoid transitory effects (5 May)
Fixed error merging training output files (5 May 2019)
Hull sliced into 100 segments -> Looks at F, P or S individually

@author: Brian Freeman and Nico van der Kolk
4 May 2019 - May the Fourth be with you
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
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

def hullextract(hull_run):
    # assume input string will be in form FS_1_126_0
    
     hull_len = len(hull_run) #get length of the string
     
     a = []
     
     # identify locations of '_' separators
     for i in range(2,hull_len):
         
         if hull_run[i] == '_':
             a.append(i)
             
    # extract values        
     hull = int(hull_run[a[0]+1:a[1]])
     fr = int(hull_run[a[1]+1:a[2]])
     leeway = int(hull_run[a[2]+1:])
    
     return hull, fr, leeway
 
def read_force(force_files):
# reads the training output files, removes the last column and 1st 3 columns
# merges files together
    
    force_n = len(force_files)
    for k in range(force_n):
        filename = force_files[k] + '.txt'
        dff = pd.read_csv(filename, skiprows=2, header = None)
        dff_n = len(dff.T)
        
        # first file (fixed 5 May 2019)
        if k == 0:
            af = dff.loc[:, dff.columns != dff_n] # remove last column
            af = (dff.loc[:, dff.columns > 2]).values # remove first  3 columns
            force_out = af # set to first file
        
        # all other files
        else:
            af = dff.loc[:, dff.columns != dff_n] # remove last column
            af = (dff.loc[:, dff.columns > 2]).values # remove first  3 columns
            force_out = np.concatenate((force_out, af), axis=1)  # add other files
    
    return force_out
#!!!!!!!!!!!!!!!!!!!!!!!! End functions !!!!!!!!!!!!!!!!!!!!!!!!!!!
    
# read input datafile
input_file = 'dKE.txt'
df = pd.read_csv(input_file, skiprows=2, header = None)

# set up input training file
n_dflen = len(df)
n_dfcols = len(df.columns)

hull_p = np.zeros((n_dflen,3))

# get hull, froud, and leeway from ID
for j in range(n_dflen):
    
    hull_p[j,0], hull_p[j,1], hull_p[j,2] = hullextract(df.iloc[j,0])
    
frnum = np.asmatrix(hull_p[:,1]).T; leewaynum = np.asmatrix(hull_p[:,2]).T
    
ID_onehot = onehot(hull_p[:,0]) # convert hull number into 1 hot code

# remove ID column and convert to numpy matrix
a = (df.loc[:, df.columns != 0]).values

# combine input datasets
dKE = np.concatenate((ID_onehot, frnum, leewaynum, a ), axis=1)
len_dKE = len(dKE.T)

# min/max input dataset
dKE_in = minmax(dKE) 

# read in training output files
#force_files = ['Fx', 'Fy', 'Px', 'Py', 'Sx', 'Sy']
force_files = ['Fx','Fx']
#force_files = ['Fy','Fy']
#force_files = ['Sx','Sy']

# prepare output training set using read_force function
force_out = minmax(read_force(force_files))
len_force = len(force_out.T)

# make complete matrix to shuffle rows
tot_train = np.concatenate((dKE_in, force_out ), axis=1)
len_tot = len(tot_train.T)

# ask if data should be shuffled
right_vars = 'y'
if right_vars == 'y':
    print('Data will be shuffled')
    shuffle = True
    
# shuffle/randomize 
if shuffle == True:
    tot_train = np.take(tot_train, 
                        np.random.permutation(tot_train.shape[0]), 
                                                     axis=0, out= tot_train)
        
# extract input and output training sets
train_in = tot_train[:,0:len_dKE]
train_out = tot_train[:,len_dKE:(len_tot-len_dKE)]

tot_set, in_shape = np.shape(train_in)
tot_set, out_shape = np.shape(train_out)

# set up training and testing values
ntrain = int(0.7* tot_set)

x_train = train_in[0:ntrain,:]
y_train = train_out[0:ntrain,:]

x_test = train_in[ntrain:tot_set,:]
y_test = train_out[ntrain:tot_set,:]

# build MLP model
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

batch = 50 # batch size

history = model.fit(train_in, train_out,
          epochs = 200,
          shuffle = True,
          verbose = 1,
          batch_size = batch,
          validation_data =(x_test, y_test))

score = model.evaluate(x_test, y_test, batch_size = batch)

#plot results
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.plot(history.history['acc'])
ax1.plot(history.history['val_acc'])
ax1.set_ylabel('Accuracy')
ax1.legend(['Train', 'Test'], loc='right')
#ax1.set_ylim([.6,1])

ax2=ax1.twinx()
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_ylabel('Loss')
#ax2.set_ylim([0,.2])

plt.title('Model Accuracy / Loss')
fig.tight_layout()
plt.show()

'''
#save results
df = pd.DataFrame(history.history)
writer = pd.ExcelWriter('dwas_mlp_output.xlsx')
df.to_excel(writer,'DWAS MLP Results')
writer.save()
'''
