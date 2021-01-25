#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:09:48 2020

@author: eric
"""


import numpy as np
import pickle


algoritmos = ['sRVFL','RVFL3','dRVFL3','edRVFL3','ELM3']
# for h in range(4):    
train,validation, test = list(),list(), list()
#Loop de los 4 horizontes estudiados t+1,t+2,t+3,t+4
for h in range(4):
    
    print(h)
    for i in range(len(algoritmos)):
        
        with open('metrics/'+algoritmos[i]+'_metrics_t+'+str(h+1)+'.pkl', 'rb') as f:
            pruebas = pickle.load(f)
        # toCSV.append(pruebas)
        
        # for horizon in range(4):
        print('\n'+algoritmos[i])
        print('Train')
        print(np.array(pruebas['train']))
        print('Validation')
        print(np.array(pruebas['val']))
        print('Test')
        print(np.array(pruebas['test']))
        train.append(np.array(pruebas['train']))
        validation.append(np.array(pruebas['val']))
        test.append(np.array(pruebas['test']))
            
    
        # plt.subplot(5,1,i+1)    
        # plt.plot(pruebas['train'], label='Train')
        # plt.plot(pruebas['val'], label = 'Validation')
        # plt.plot(pruebas['test'], label= 'Test')
        # plt.legend()
        # plt.ylabel('Score')
        # plt.xlabel('Target location at Madrid')
        # plt.set_ylim(0.5,1)
            
import csv
b = open('results_randomized_PeMS_train.csv', 'w')
a = csv.writer(b)
a.writerows(train)
b.close()
b = open('results_randomized_PeMS_validation.csv', 'w')
a = csv.writer(b)
a.writerows(validation)
b.close()
b = open('results_randomized_PeMS_test.csv', 'w')
a = csv.writer(b)
a.writerows(test)
b.close()
