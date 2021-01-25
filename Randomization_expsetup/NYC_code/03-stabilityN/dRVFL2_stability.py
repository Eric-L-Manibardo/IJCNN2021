#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:47:33 2020

@author: eric
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from skRVFL import deepRVFLRegressor

samples_day = 271
days_week   = 7
samples_week = samples_day*days_week
months_year = 12


    
##############################################
#############      START     #################
##############################################
    
#Architecture parameters
batch_size = 672 #muestras en 1 semana
n_timesteps = 5
n_features = 1 # usamos una sola espira

#empty dict for storing final results
results = {}
#empty dataframe
df = pd.DataFrame()

num_test=100
neurons = [1,10,50,100,500,1000]
# neurons = [1,5,10]
espiras = ['145','418','137','169','157','259','217','106','295','318']
for h in range(4):    
    #Loop de los 4 horizontes estudiados t+1,t+2,t+3,t+4
      
    
    for k in range(len(espiras)):
        
        # load TRAIN
        df_train = pd.read_csv('dataset_TRAIN_NYC/'+ espiras[k]+'train_NYC.csv') 
        #Pasamos a array target value y features
        y_train = df_train['target'].values 
        X_train = StandardScaler().fit_transform(df_train.iloc[:,1:6].values)
        # load TEST
        df_test = pd.read_csv('dataset_TEST_NYC/'+ espiras[k]+'test_NYC.csv') 
        #Pasamos a array target value y features
        y_test = df_test['target'].values 
        X_test = StandardScaler().fit_transform(df_test.iloc[:,1:6].values)
        
        #adjust forecasting horizon  
        if h==0:
            y_train = y_train
            y_test = y_test
            X_train = X_train
            X_test = X_test
            
        else:            
            y_train = y_train[h:]
            y_test = y_test[h:]
            X_train = X_train[:-h]
            X_test = X_test[:-h]
      
         
        for n in range(len(neurons)):
            # reset test values for one architecture
            test_loss = list()  
            for t in range(num_test):
                # set RVFL hiperparams
                model = deepRVFLRegressor(neuronsPerLayer=(
                    neurons[n],neurons[n]))  
                #fit model
                model.fit(X_train, y_train)
            
                #test metrics for a set of neurons, one location, one forecasting horizon = 6 * 10 * 4
                pred = model.predict(X_test)
                test_loss.append(r2_score(y_test, pred))
                
            # results[espiras[k]] = np.array(test_loss)
            results['a'+str(neurons[n])+'e' + espiras[k] + 'h'+ str(h+1)] = np.array(test_loss)
            
#Store metrics on Dataframe
metrics_per_h = pd.DataFrame(data=results)
#Set index names for hyperparams configurations
# for i in range(len(neurons)):
#     metrics_per_h = metrics_per_h.rename(index={i:str(neurons[i])})
df =df.append(metrics_per_h)
        



#store results dataframe
df.to_csv('stability/dRVFL2_stability.csv')

   
print('Closing script...')

