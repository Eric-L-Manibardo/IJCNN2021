#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:47:33 2020

@author: eric
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import os
import pickle


samples_day = 271
days_week   = 7
samples_week = samples_day*days_week
months_year = 12

def config_device(computing_device):
    if 'gpu' in computing_device:
        device_number = computing_device.rsplit(':', 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_number
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def r2_train(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def build_FNN(batch_size, n_timesteps, n_features, h1, h2, h3):
    ### input ###
    input1 = keras.layers.Input(shape=(n_timesteps,))
    hidden1 = keras.layers.Dense(int(h1), activation='relu')(input1)
    hidden2 = keras.layers.Dense(int(h2), activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(int(h3), activation='relu')(hidden2)
    output = keras.layers.Dense(1)(hidden3)
    model = keras.models.Model(inputs=input1, outputs=output)
     
    model.compile(loss='mse', optimizer= 'adam',metrics=[r2_train])    
    return model

    
##############################################
#############      START     #################
##############################################
#Architecture parameters
config_device('gpu:3')
epoch = 150
batch_size = 672 #samples per week
n_timesteps = 5
n_features = 1 # as we use only one traffic reader for input

# Traffic reader IDs
espiras = ['145','418','137','169','157','259','217','106','295','318']
#Loop about 4 studied forecasting horizons t+1,t+2,t+3,t+4
for h in range(4):    
    val_loss, train_loss, test_loss = list(), list(), list()
    with open('FNN_trials_t+'+str(h+1)+'.pkl', 'rb') as f:
        pruebas = pickle.load(f)
    
    for k in range(len(espiras)):
        # load train Dataset
        df_train = pd.read_csv('dataset_TRAIN_NYC/'+ espiras[k]+'train_NYC.csv') 
        y_train = df_train['target'].values 
        X_train = StandardScaler().fit_transform(df_train.iloc[:,1:6].values)
        X_train = X_train.reshape(X_train.shape[0], n_timesteps, n_features)
        # load test Dataset
        df_test = pd.read_csv('dataset_TEST_NYC/'+ espiras[k]+'test_NYC.csv')
        y_test = df_test['target'].values 
        X_test = StandardScaler().fit_transform(df_test.iloc[:,1:6].values)
        X_test = X_test.reshape(X_test.shape[0], n_timesteps, n_features)
          
        #special t+1 format
        if h==0:
            y_train = y_train
            y_test = y_test
            X_train = X_train
            X_test = X_test
        #for the rest of forecasting horizons   
        else:            
            y_train = y_train[h:]
            y_test = y_test[h:]
            X_train = X_train[:-h]
            X_test = X_test[:-h]
        
        #Best hyperparam config
        best = pruebas[k].best_trial    
        params = best['misc']['vals']
        
        model = build_FNN(batch_size, n_timesteps, n_features, np.array(params['h1']), np.array(params['h2']), np.array(params['h3']))
        history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, shuffle=True, verbose =0)
        #Model metrics
        pred = model.predict(X_test)
        test_loss.append(r2_score(y_test, pred))
        train_loss.append(history.history['r2_train'][-1])
        val_loss.append(best['result']['loss']*-1)
    
    #Store metrics
    metrics = {}
    metrics['train'] = train_loss
    metrics['val'] = val_loss
    metrics['test'] = test_loss
    with open('FNN_metrics_t+'+str(h+1)+'.pkl', 'wb') as f:
        pickle.dump(metrics, f)
        

    
print('Closing script...')

