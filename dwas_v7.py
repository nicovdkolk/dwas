# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:52:54 2020

Updated dwasv6_2_2.py to scale by minmax instead of normalizing

@author: Brian
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Nadam, Adam
from keras.models import model_from_json
import time


def make_trig(df):
    # takes one df column and makes np sine & cosine columns
    rads = np.asmatrix(np.radians(df.values))
    cos = np.cos(rads).T
    sin = np.sin(rads).T
    return np.concatenate((cos,sin), axis = 1)

'''# **********************************************************************
# Main program start
'''
       
# fix random seed for reproducibility
np.random.seed(1)
Xbatch = []
Xlookback= []
Xdropout= []

# get data files
hull_file = 'hulls.txt'  # hulls
dwas_file = 'DWA_DATA_ND_OCT.txt' #main dwas data with hull number to link to hull_file

#upload data & prediction files
hulls_df = pd.read_csv(hull_file, engine = 'python', sep= '\s+|\t+|\s+\t+|\t+\s+')
dwas_df = pd.read_csv(dwas_file)

#add hull info to each training observation
df_tot1 = pd.merge(right=dwas_df,left=hulls_df, 
                  how='left', 
                  left_on='ID', 
                  right_on='Hull').dropna(axis=1)  #and drop na columns

df_tot1 =df_tot1.drop(['Description', 'Hull'], axis=1) # delete columns

df_tot1 = df_tot1.rename(columns={"ID": "Hull"}) # change ID column to Hull

###### Select data to train on by choosing 1 or all hull groups
print('Select groups to train on: ')
print('Group 0: Hulls 1-33 ')
print('Group 1: Hulls 34-44 ')
print('Group 2: Hulls 45-60 ')
print('Group 3: All Hulls')
try:
    group_select = int(input('Select Group Number (0-3) (default = 3) '))
except:
    group_select = 3 # select all hulls

group_name = 'Group_' + str(group_select)

print('\n' + group_name + ' selected..')
    
if group_select != 3:
    df_tot = df_tot1.loc[df_tot1['Group'] == group_select]
    
if group_select == 3:
    df_tot = df_tot1.copy(deep = True)    

## shuffle data
df_tot = df_tot.sample(frac=1).reset_index(drop=True)  #shuffle data

# segragate input training (x) from output training (y)
df_y = df_tot[['Xnd', 'Ynd', 'Nnd']]

# drop unneccesary rows for x - will remake leeway & heel as trig components
df_x = df_tot.drop(['Hull', 'Group','leeway', 'heel','Xnd', 'Ynd', 'Nnd'], axis=1) 

df_x = df_x.replace(np.nan,0) #convert any nan's to 0
    
# convert to numpy 
x_raw = df_x.values  
y_raw = np.asmatrix((df_y.values)) * 100 # shift decimal point to account for low values

# add leeway and trig components to x
trig = np.concatenate((make_trig(df_tot['leeway']), make_trig(df_tot['heel'])), axis = 1 )
x_raw = np.concatenate((x_raw, trig), axis = 1)

x_raw_ct = len(x_raw.T)
y_raw_ct = len(y_raw.T)

# prepare output training set using read_force function
# set up min/max scalers for input and output data
scaler_in = MinMaxScaler()
scaler_out = MinMaxScaler()

scaler_in.fit(x_raw)
scaler_out.fit(y_raw) # fit the output data

x_scale = scaler_in.transform(x_raw)
y_scale = scaler_out.transform(y_raw)

## make total training set
tot_train = np.concatenate((x_scale, y_scale), axis = 1) # merge the total data
                           
len_force = len(tot_train.T)

# make complete matrix to shuffle rows
len_tot = len(tot_train.T)

len_dKE = len(x_raw)
    
# extract input and output training sets
train_in = tot_train[:,0:x_raw_ct]
train_out = tot_train[:,x_raw_ct:((y_raw_ct+x_raw_ct))]

#get parameters for MLP nodes
tot_set, in_shape = np.shape(train_in)
tot_set, out_shape = np.shape(train_out)

# set up training and testing values
ntrain = int(0.85 * tot_set)

x_train = train_in[0:ntrain,:]
y_train = train_out[0:ntrain,:]

x_test = train_in[ntrain:tot_set,:]
y_test = train_out[ntrain:tot_set,:]

n = in_shape * 6

#**********************************************************
# build MLP model
#**********************************************************

start = time.time()
drop = .01 # dropout percentage
model = Sequential()
act_func = 'relu'

model = Sequential()
model.add(Dense(n, activation= act_func, use_bias = True, input_dim=in_shape))
model.add(Dropout(drop))
model.add(Dense(n, use_bias = True, activation= act_func))
model.add(Dropout(drop))
model.add(Dense(n, use_bias = True, activation= act_func))
model.add(Dropout(drop))
model.add(Dense(out_shape, use_bias = True, activation='tanh'))

m = 4 # of layers

#sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=False)
nadam = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
#adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)

model.compile(loss='mean_squared_error',
              optimizer = nadam,
              metrics=['accuracy'])

# print model statistics
#model.summary()

batch = 8 # batch size

#####train network
history = model.fit(x_train, y_train,
          epochs = 200,
          shuffle = False,
          verbose = 1,
          batch_size = batch, validation_split = 0.1)

score = model.evaluate(x_test, y_test, batch_size = batch)

end = time.time() 
        
#******* predict input file by transforming the data for computation then inversing the transform
# add leeway and trig components to x
'''
x_predict = (predict_df.drop(['leeway', 'heel'], axis=1)/df_max).values
    
pred_trig = np.concatenate((make_trig(predict_df['leeway']), make_trig(predict_df['heel'])), axis = 1 )
x_predict = np.concatenate((x_predict, pred_trig), axis = 1)

future_trg = model.predict(x_test) # predict output
future_trg_xfrm = np.abs(scaler_out.inverse_transform(future_trg)/100)

# force 0 when leeway & heel are 0
for i in range(len(x_test)):
    if predict_df.leeway[i] == 0 and predict_df.heel[i] == 0:
        future_trg_xfrm[i,1] = 0
        future_trg_xfrm[i,2] = 0
'''        

print('\nTotal training time = ', round((end-start)/60,1), 'minutes')

# find error of each force by comparing test data set        
future_err = model.predict(x_test)

y_test_xfrm = scaler_out.inverse_transform(y_test)/100
future_err_xfrm = np.abs(scaler_out.inverse_transform(future_err))/100

# take the mean absolute percentage error (MAPE) of each force component
err = np.abs(np.round(100*(y_test_xfrm - future_err_xfrm)/y_test_xfrm,1))
for j in range(len(err.T)):
    err_col = err.T[j]
    print(round(err_col[np.isfinite(err_col)].mean(),1),end = '%, ')
    
#******************* save model and weights **************
try:
    save_wghts = input('Save weights and model? (default = n) ')
except:
    save_wghts = 'n'
    
if save_wghts == 'y':
    
    ## save MinMaxScaler results
    scaler_in_file = 'scaler_in_' + group_name + '.save'
    scaler_out_file = 'scaler_out_' + group_name + '.save'
    
    joblib.dump(scaler_in, scaler_in_file) 
    joblib.dump(scaler_out, scaler_out_file) 

    # serialize forecast (FC) model to JSON
    model_json = model.to_json()
    with open('dwas_'+ group_name +'_model.json', 'w') as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('dwas_' + group_name + '_weights.h5')
        print("\nSaved model and weights to disk")

#plot training, validation and loss results
'''
fig = plt.figure(1, (7,4))
ax = fig.add_subplot(1,1,1)

ax.plot(history.history['accuracy'], label='Training Accuracy')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # convert y axis to %
plt.title('Model accuracy ('+str(n)+' nodes, '+str(m)+' layers)')
ax.set_ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='center right')
ax2 = ax.twinx()
ax2.plot(history.history['loss'], color = 'green', label='Loss')
ax2.set_ylabel('Loss')
#ax2.set_yscale('log')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc='center right')
plt.show()
'''

##### plot baseline and predictions
#n_plot = 10 #random.randint(0,len(testY)-250)
plt.close('all')
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)

ax1.set_title(group_name + ' Xnd')
ax1.plot(y_test_xfrm[:,0])
ax1.plot(future_err_xfrm[:,0])

ax2.set_title(group_name + ' Ynd')
ax2.plot(y_test_xfrm[:,1])
ax2.plot(future_err_xfrm[:,1])

ax3.set_title(group_name + ' Nnd')
ax3.plot(y_test_xfrm[:,2])
ax3.plot(future_err_xfrm[:,2])
ax3.set_xlabel('Observation')

plt.tight_layout()
plt.show()




