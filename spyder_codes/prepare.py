#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:12:46 2019

@author: doctor
"""

import os,sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from datetime import datetime
from pandas import concat
import pandas as pd
import random
class PrepareDataset():

    def ReadDatas():
        
#        dataset = pd.read_excel('/home/doctor/Desktop/batarya/profiles.xlsx',  parse_dates = [['Tarih', 'Saat']], index_col=0)
#        dataset.rename(columns={'Index':'Index',
#                          'Seri No':'serial_no',
#                          'Aktif Enerji Saatlik':'active_power'}, 
#                 inplace=True)
#        dataset.index.name = 'Index'
#        # mark all NA values with 0
#        dataset['active_power'].fillna(0, inplace=True)
#        dataset['temp']= [random.randint(25,30) for i in range(0,len(dataset['serial_no']))]
#        dataset['humidity']= [random.randint(10,20) for i in range(0,len(dataset['serial_no']))]
#        dataset['condition']= [random.choice(['cloudy','sunny','rainy','clear']) for i in range(0,len(dataset['serial_no']))]
#        # drop the first 24 hours
#        #dataset = dataset[24:]
#        # summarize first 5 rows
#        print(dataset.head(5))
#        # save to file
#        dataset.to_csv('/home/doctor/Desktop/batarya/dataset_ready.csv')
        dataset= pd.read_csv('/home/doctor/Desktop/batarya/dataset_ready.csv', index_col=0,dtype={'active_power': str})
        dataset.sort_index(inplace=True)
        return dataset
    
    
    def ReadSQL():
        pass
    
    def SupervisorNormalization3D(dataset, n_hours, n_features):
        
                # convert series to supervised learning
        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        	n_vars = 1 if type(data) is list else data.shape[1]
        	df = DataFrame(data)
        	cols, names = list(), list()
        	# input sequence (t-n, ... t-1)
        	for i in range(n_in, 0, -1):
        		cols.append(df.shift(i))
        		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        	# forecast sequence (t, t+1, ... t+n)
        	for i in range(0, n_out):
        		cols.append(df.shift(-i))
        		if i == 0:
        			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        		else:
        			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        	# put it all together
        	agg = concat(cols, axis=1)
        	agg.columns = names
        	# drop rows with NaN values
        	if dropnan:
        		agg.dropna(inplace=True)
        	return agg
        
#        dataset=dataset.drop(columns= ['serial_no'])
#        dataset.sort_index(inplace=True)
        values = dataset.values
        # integer encode direction
#        encoder = LabelEncoder()
#        values[:,3] = encoder.fit_transform(values[:,3])
      
        # ensure all data is float
        values = values.astype('float32')
        
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        
                
        # specify the number of lag hours
        
        
        # frame as supervised learning
        reframed = series_to_supervised(scaled, n_hours, 1)
        print(reframed.shape)   
         
        # split into train and test sets
        values = reframed.values
        n_train_hours = 300*24
        train = values[:n_train_hours, :]
        test = values[n_train_hours-3:, :]
        
        n_obs = n_hours * n_features
        LSTM_test_24 = values[-24:, :n_obs]
        LSTM_test_24 = LSTM_test_24.reshape((LSTM_test_24.shape[0], n_hours, n_features))
        # split into input and outputs
        n_obs = n_hours * n_features
        LSTM_train_X, LSTM_train_y = train[:, :n_obs], train[:, -n_features]
        LSTM_test_X, LSTM_test_y = test[:, :n_obs], test[:, -n_features]
        print(LSTM_train_X.shape, len(LSTM_train_X), LSTM_train_y.shape)
        
        
        # reshape input to be 3D [samples, timesteps, features]
        LSTM_train_X = LSTM_train_X.reshape((LSTM_train_X.shape[0], n_hours, n_features))
        LSTM_test_X = LSTM_test_X.reshape((LSTM_test_X.shape[0], n_hours, n_features))
        print(LSTM_train_X.shape, LSTM_train_y.shape, LSTM_test_X.shape, LSTM_test_y.shape)
        
        return LSTM_train_X,LSTM_train_y,LSTM_test_X,LSTM_test_y,scaler,LSTM_test_24

    def ConvertandNormalization2D(dataset):
        
    
#        dataset=dataset.drop(columns= ['serial_no'])
        # summarize first 5 rows
        print(dataset.head(5))
        dataset=dataset[['temp','humidity','active_power','active_power']]
        dataset.columns = ['temp','humidity','active_power','active_power_y']
#        dataset.sort_index(inplace=True)
    
        encoder = LabelEncoder()
#        dataset['condition'] = encoder.fit_transform(dataset['condition'])
        
        dates_float=dataset.index[-24:]
        
        df_24=dataset.tail(24)     
        x_24_real=df_24.iloc[:,0:3]
        y_24_real=df_24.iloc[:,3]
        values = dataset.values
        # integer encode direction
        encoder = LabelEncoder()
#        values[:,2] = encoder.fit_transform(values[:,2])
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        
        x=scaled[0:8760-24,0:3]
        y=scaled[0:8760-24,3].reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
        
        sx=MinMaxScaler()
        sy=MinMaxScaler()
        y_normal_24=sy.fit_transform(np.array(y_24_real).reshape(-1,1))
        x_normal_24=sx.fit_transform(x_24_real)
        print("Dataset is Ready!")
        
        return x_train, x_test, y_train, y_test,x_normal_24,y_normal_24,x_24_real,y_24_real,dates_float,sx,sy
            

        
        
        