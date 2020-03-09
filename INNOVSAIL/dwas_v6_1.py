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
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Nadam, Adam

'''
def minmax(a):
    scaler = MinMaxScaler()
    scaler.fit(a)
    return scaler.transform(a)
'''

def make_trig(df):
    # takes one df column and makes np sine & cosine columns
    rads = np.asmatrix(np.radians(df.values))
    cos = np.cos(rads).T
    sin = np.sin(rads).T
    return np.concatenate((cos,sin), axis = 1)
    
# **********************************************************************
# Main program start
       
# fix random seed for reproducibility
np.random.seed(1)
Xbatch = []
Xlookback= []
Xdropout= []

# get data files
hull_file = 'hulls.txt'  # hulls
dwas_file = 'DWA_DATA_ND_RECT.txt' #main dwas data with hull number to link to hull_file

# $$$$$$$$$$$$$$ prediction file - prediction file has hull info
predict_file = 'hull1.xlsx'

current_path = os.getcwd()  # get current working path to save later

#upload data & prediction files
hulls_df = pd.read_csv(hull_file, engine = 'python', sep= '\s+|\t+|\s+\t+|\t+\s+')
dwas_df = pd.read_csv(dwas_file)
predict_df = pd.read_excel(predict_file)

#add hull info to each training observation
df_tot = pd.merge(right=dwas_df,left=hulls_df, 
                  how='left', 
                  left_on='ID', 
                  right_on='Hull').dropna(axis=1)  #and drop na columns

df_tot =df_tot.drop(['Description', 'Hull'], axis=1) # delete columns

df_tot = df_tot.rename(columns={"ID": "Hull"}) # change ID column to Hull

# segragate input training (x) from output training (y)
df_y = df_tot[['Xnd', 'Ynd', 'Nnd']]

# drop unneccesary rows for x - will remake leeway & heel as trig components
df_x = df_tot.drop(['Hull', 'leeway', 'heel','Xnd', 'Ynd', 'Nnd'], axis=1) 

# convert to numpy 
x_raw = df_x.values  
y_raw = np.asmatrix((df_y.values)) * 100 # shift decimal point

# add leeway and trig components to x
trig = np.concatenate((make_trig(df_tot['leeway']), make_trig(df_tot['heel'])), axis = 1 )
x_raw = np.concatenate((x_raw, trig), axis = 1)

x_raw_ct = len(x_raw.T)
y_raw_ct = len(y_raw.T)

# prepare output training set using read_force function
# set up min/nax scaler with total data set for input and output
scaler_in = MinMaxScaler()
scaler_out = MinMaxScaler()

scaler_in.fit(x_raw)  # fit the input data
scaler_out.fit(y_raw) # fit the output data

force_out = np.concatenate((scaler_in.transform(x_raw), 
                            scaler_out.transform(y_raw)), axis = 1) # scale the total data

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
        
    # Confirm model and continue or cancel
    try:
        right_vars = input('Continue with modeling (y/n)? (Default = y): ')
    except:
        right_vars = 'y'
        
    if right_vars == 'n':
        sys.exit()
        
# extract input and output training sets
train_in = tot_train[:,0:x_raw_ct]
train_out = tot_train[:,x_raw_ct:((y_raw_ct+x_raw_ct))]

#get parameters for MLP nodes
tot_set, in_shape = np.shape(train_in)
tot_set, out_shape = np.shape(train_out)

# set up training and testing values
ntrain = int(0.85* tot_set)

x_train = train_in[0:ntrain,:]
y_train = train_out[0:ntrain,:]

x_test = train_in[ntrain:tot_set,:]
y_test = train_out[ntrain:tot_set,:]

n = in_shape * 4

drop = .01 # dropout percentage
#**********************************************************
# build MLP model
#**********************************************************

model = Sequential()
act_func = 'relu'

model = Sequential()
model.add(Dense(n, activation= act_func, use_bias = True, input_dim=in_shape))
model.add(Dropout(drop))
model.add(Dense(n, use_bias = True, activation= act_func))
model.add(Dropout(drop))
model.add(Dense(out_shape, use_bias = True, activation='tanh'))

m = 3 # of layers

sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=False)
nadam = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)

model.compile(loss='mean_squared_error',
              optimizer = adam,
              metrics=['accuracy'])

# print model statistics
model.summary()

batch = 8 # batch size

#####train network
history = model.fit(x_train, y_train,
          epochs = 100,
          shuffle = False,
          verbose = 1,
          batch_size = batch, validation_split = 0.15)

score = model.evaluate(x_test, y_test, batch_size = batch)

# predict input file by transforming the data for computation then inversing the transform
# add leeway and trig components to x

x_predict = predict_df.drop(['leeway', 'heel'], axis=1).values
pred_trig = np.concatenate((make_trig(predict_df['leeway']), make_trig(predict_df['heel'])), axis = 1 )
x_predict = np.concatenate((x_predict, pred_trig), axis = 1)

future_trg = model.predict(scaler_in.transform(x_predict))
future_trg_xfrm = np.abs(scaler_out.inverse_transform(future_trg)/100)

# force 0 when leeway & heel are 0
for i in range(len(predict_df)):
    if predict_df.leeway[i] == 0 and predict_df.heel[i] == 0:
        future_trg_xfrm[i,1] = 0
        future_trg_xfrm[i,2] = 0
        

# find error of each force by comparing test data set        
future_err = model.predict(x_test)

y_test_xfrm = scaler_out.inverse_transform(y_test)/100
future_err_xfrm = np.abs(scaler_out.inverse_transform(future_err))/100

# take the mean absolute percentage error (MAPE) of each force component
err = np.abs(np.round(100*(y_test_xfrm - future_err_xfrm)/y_test_xfrm,1))
for j in range(len(err.T)):
    err_col = err.T[j]
    print(round(err_col[np.isfinite(err_col)].mean(),1),end = '%, ')

#plot results

fig = plt.figure(1, (7,4))
ax = fig.add_subplot(1,1,1)

ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
#ax.plot(history.history['loss'])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # convert y axis to %
plt.title('Model accuracy ('+str(n)+' nodes, '+str(m)+' layers)')
plt.ylabel('Percentage')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='center right')
plt.show()


if 1:
    df = pd.DataFrame(history.history)
    writer = pd.ExcelWriter('dwas_mlp_output.xlsx')
    df.to_excel(writer,'DWAS MLP Results')
    writer.save()
