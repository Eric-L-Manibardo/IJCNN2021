#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:23:01 2021

@author: eric
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection


# available options for file name
model = ['RVFL','dRVFL','edRVFL','ELM'] 
hidden = ['2','4','6','8','10'] #hidden layers

# available options for header || example of header a50e402577h1
neurons = ['1','10','50','100','500','1000']
espiras = ['402577','402045','401952','407479','404356','401256','401657','407990','400359','409306']
horizon = ['1','2','3','4']




# store values to plot || order model_L_neurons
mean, std = list(),list()
for m in range(len(model)):
    for h in range(len(hidden)):
        results = pd.read_csv('stability/'+model[m]+hidden[h]+'_stability.csv')
        for n in range(len(neurons)):
            mean.append(results['a'+neurons[n]+'e'+espiras[9]+'h'+horizon[0]].mean())
            std.append(results['a'+neurons[n]+'e'+espiras[9]+'h'+horizon[0]].std())
         
# convert to array
y = np.array(mean)
x = np.arange(len(y))
std = np.array(std)

# Create line segments: 1--2, 2--17, 17--20, 20--16, 16--3, etc.
segments_x = np.r_[x[0], x[1:-1].repeat(2), x[-1]].reshape(-1, 2)
segments_y = np.r_[y[0], y[1:-1].repeat(2), y[-1]].reshape(-1, 2)

# Create conditional thresholds for color assigment
ca = list()
for i in range(len(hidden)):
    for j in range(len(model)):
        for k in range(len(neurons)):
            if k==5: ca.append('blank')
            else: ca.append(model[j])

# Assign colors to the line segments
linecolors = list()
for ca_ in ca:
    if ca_=='RVFL':
        linecolors.append('blue')
    elif ca_=='dRVFL':
        linecolors.append('red')
    elif ca_=='edRVFL':
        linecolors.append('green')
    elif ca_=='ELM':
        linecolors.append('black')
    else:
        linecolors.append('white')

lines = list()    
for i in range(len(segments_x)):
    # lines.append([tuple(segments_x[i]),tuple(segments_y[i])])
    lines.append(
        [tuple([segments_x[i,0],segments_y[i,0]]),
         tuple([segments_x[i,1],segments_y[i,1]])])
    

        

        
# Create figure
plt.figure()
ax1 = plt.axes()

# Add the mean values of the R2 score
ax1.add_collection(LineCollection(lines, colors=linecolors))
ax1.autoscale()

# Show +std and -std
# for i in range(len(hidden)):
#     for j in range(len(model)):
#         for k in range(len(neurons)):
#             if k<5: 
                # ax1.fill_between(range(6)+i+j,y-std,y+std,alpha=0.5) 
            
ax1.fill_between(range(len(y)),y-std,y+std,alpha=0.5) 
# ax1.fill_between(range(0,6*1),y[0:6]-std[0:6],y[0:6]+std[0:6],alpha=0.5) 
# ax1.fill_between(range(6,6*2),y[0:6]-std[0:6],y[0:6]+std[0:6],alpha=0.5,color='red') 

ax1.set_title('PeMS_espira_'+espiras[3]+'_horizonte_'+horizon[0])
ax1.set_ylabel('R² score (mean & standard deviation)')
ax1.set_xlabel('Number of neurons')

# create custom x tick values
x_labels = list()
count = 0
for i in range(len(y)):
    
    x_labels.append(neurons[count])
    count = count +1
    if count == len(neurons): count = 0      


plt.xticks(x, x_labels,rotation=60)

ax2 = ax1.twiny()
ax2.set_xticks(range(20),range(20))
        
        
        
        
        
        
        
        
        