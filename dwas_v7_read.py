# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 23:35:34 2020

Reads 4 files:
    *.JSON for model architecture
    *.h5 for model weights
    *_input.save and *_output.save for minmax scaling of inputs/outputs
    
File names should be in format *_Group_*.*

Input must be in the following order with column names:
['Length',  'Beam', 'Draft', 'Cp', 'Cb', 'Cm', 'L_B', 'B_T', 'T_L', 'L_vol3', 
 'Cwp', 'AwpSw', 'Rb_T', 'Deadrise', 'Fn', 'leeway', 'heel']

@author: Brian
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Nadam, Adam
from keras.models import model_from_json

def make_trig(df):
    # takes one df column and makes np sine & cosine columns
    rads = np.asmatrix(np.radians(df.values))
    cos = np.cos(rads).T
    sin = np.sin(rads).T
    return np.concatenate((cos,sin), axis = 1)

def make_model(group_name):
    
    ## load scaling values
    scaler_in_file = 'scaler_in_' + group_name + '.save'
    scaler_out_file = 'scaler_out_' + group_name + '.save'
    
    scaler_in = joblib.load(scaler_in_file)
    scaler_out = joblib.load(scaler_out_file)
    
     #**********************************************************
    # build pre-trained MLP model
    #**********************************************************
    
    # ***** load json and weight files to re-create model
    
    json_file = open('dwas_' + group_name + '_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # load weights into new model
    
    model.load_weights('dwas_' + group_name + '_weights.h5')
    
    nadam = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error',
                  optimizer = nadam,
                  metrics=['accuracy'])
    return model, scaler_in, scaler_out

'''
##------------------------
Begin main
'''
### only need to change this for input names:

group_name = 'Group_3'

## load pre-formated (but not scaled) data for evaluating
## change input data file as necessary - but format must be the same!

predict_file = 'hull1.xlsx'
df_tot = pd.read_excel(predict_file)

##########
order_list = ['Length',  'Beam', 'Draft', 'Cp', 'Cb', 'Cm', 'L_B', 
              'B_T', 'T_L', 'L_vol3', 'Cwp', 'AwpSw', 'Rb_T', 'Deadrise', 
              'Fn', 'leeway', 'heel']

## build model and scaling values
model, scaler_in, scaler_out = make_model(group_name)

## if the column headers are not in the right order, exit the program
if order_list != list(df_tot):
    print('\nInput column headers are not correct. Terminating...')
    sys.exit()
    
# drop unneccesary rows for x - will remake leeway & heel as trig components
df_x = df_tot.drop(['leeway', 'heel'], axis=1) 

# convert to numpy 
x_raw = df_x.values  

# add leeway and trig components to x
trig = np.concatenate((make_trig(df_tot['leeway']), make_trig(df_tot['heel'])), axis = 1 )
x_raw = np.concatenate((x_raw, trig), axis = 1)

x_raw_ct = len(x_raw.T)

# scale input values
x_test = scaler_in.transform(x_raw)

future_trg = model.predict(x_test) # predict output
df_out = pd.DataFrame(np.abs(scaler_out.inverse_transform(future_trg)/100))
df_out.columns = ['X', 'Y', 'N']

## merge input file with outout forces
df_final = pd.concat((df_out, df_tot), axis = 1)




