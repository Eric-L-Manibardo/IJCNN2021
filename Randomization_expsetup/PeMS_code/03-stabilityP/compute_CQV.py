#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:23:01 2021

@author: eric
"""


import pandas as pd


# available options for file name
model = ['RVFL','dRVFL','edRVFL','ELM'] 
hidden = ['2','4','6','8','10'] #hidden layers

# available options for header || example of header a50e402577h1
neurons = ['1','10','50','100','500','1000']
espiras = ['402577','402045','401952','407479','404356','401256','401657','407990','400359','409306']
horizon = ['1','2','3','4']

def compute_QCD(df):
    #compute the Quartile Coefficient of Dispersion (Q3-Q1)/(Q3+Q1)
    QCD = (df.quantile(0.75)-df.quantile(0.25)) / (df.quantile(0.75)+df.quantile(0.25))
    return QCD
    

QCD_matrix = pd.DataFrame()

for ho in range(len(horizon)):
    for e in range(len(espiras)):
        # empty dic
        metrics = {}
        # store values of one loop and one horizon to plot || order model_L_neurons
        # mean, std, res = list(),list(), list()
        for m in range(len(model)):
            for h in range(len(hidden)):
                results = pd.read_csv('stability/'+model[m]+hidden[h]+'_stability.csv')
                for n in range(len(neurons)):
                    
                    # Compute QCD for certain neuron, loop, horizon, model and hidden layer
                    QCD = compute_QCD(results['a'+neurons[n]+'e'+espiras[e]+'h'+horizon[ho]])
                    # res.append(QCD)
                    metrics['n'+neurons[n]+'L'+hidden[h]+model[m]] = QCD
                    
        # QCD_matrix = pd.Series(metrics).to_frame('e' + espiras[0] + 'h'+ str(0))
        QCD_matrix['e' + espiras[e] + 'h'+ horizon[ho]] = pd.Series(metrics)


QCD_matrix.to_excel('CQV_PeMS.xlsx')


print('Closing script...')



        
        
        
        
        
        
        
        