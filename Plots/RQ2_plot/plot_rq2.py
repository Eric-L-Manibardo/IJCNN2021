#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:00:23 2021

@author: eric
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set up the matplotlib figure
sns.despine(left=True)
sns.set(style="whitegrid")

#font size
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



neurons = ['1','10','50','100','500','1000'] # neurons per layer
hidden = ['2','4','6','8','10'] #hidden layers
# palette_custom = ["#FF595E", "#FFCA3A" , "#8AC926", "#1982C4",'#6A4C93']
palette_custom = ["#CB2C2A", "#EA9010" , "#6A9A1D", "#1982C4",'#6A4C93']
city = ['Madrid', 'PeMS', 'NYC','Seattle']

def prepare_data(city):
    df = pd.read_excel('CQV_'+city+'.xlsx')
    # extract one horizon only
    h=0
    df = df.iloc[:, h*10 :h*10 + 11]
    
    # extract 3 30 first values for RVFL and 0 last values for ELM
    rvfl = df.iloc[:30,: ]
    elm = df.iloc[90:120,:]
    
    # compute the mean per row
    rvfl['CQV'] = df.mean(axis=1)
    elm['CQV'] = df.mean(axis=1)
    
    # negative CQV values implies negative R2 values, so in these cases maximum CQV is set as 1
    rvfl.CQV[rvfl.CQV < 0 ] = 1
    elm.CQV[elm.CQV < 0 ] = 1
    
    # set the minimum possible value as 0.00001
    minimum_val = 0.00001
    rvfl.CQV[rvfl.CQV < minimum_val ] = minimum_val
    elm.CQV[elm.CQV < minimum_val ] = minimum_val
    
    # add x labels to dataframe
    n = list()
    for i in range(len(hidden)):
        for j in range(len(neurons)):
            n.append(neurons[j])
    rvfl['Neurons per layer']= n
    elm['Neurons per layer']= n
    
    return rvfl, elm

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
(ax1, ax2), (ax3, ax4) = axs
fig.suptitle('CQV of RVFL (continuous) and ELM (dashed) models', y=0.95,x=0.3)


#load data
rvfl, elm = prepare_data(city[0])
for i in range(len(hidden)):
    ax1.semilogy(rvfl.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], rvfl.iloc[6*i : 6*i + 6,: ]['CQV'],label = hidden[i],color=palette_custom[i])
    ax1.semilogy(elm.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], elm.iloc[6*i : 6*i + 6,: ]['CQV'],color=palette_custom[i],linestyle="dashed")

rvfl, elm = prepare_data(city[1])
for i in range(len(hidden)):
    ax2.semilogy(rvfl.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], rvfl.iloc[6*i : 6*i + 6,: ]['CQV'],label = hidden[i],color=palette_custom[i])
    ax2.semilogy(elm.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], elm.iloc[6*i : 6*i + 6,: ]['CQV'],color=palette_custom[i],linestyle="dashed")
    
rvfl, elm = prepare_data(city[2])   
for i in range(len(hidden)):
    ax3.semilogy(rvfl.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], rvfl.iloc[6*i : 6*i + 6,: ]['CQV'],label = hidden[i],color=palette_custom[i])
    ax3.semilogy(elm.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], elm.iloc[6*i : 6*i + 6,: ]['CQV'],color=palette_custom[i],linestyle="dashed")
    
rvfl, elm = prepare_data(city[3])
for i in range(len(hidden)):
    ax4.semilogy(rvfl.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], rvfl.iloc[6*i : 6*i + 6,: ]['CQV'],label = hidden[i],color=palette_custom[i])
    ax4.semilogy(elm.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], elm.iloc[6*i : 6*i + 6,: ]['CQV'],color=palette_custom[i],linestyle="dashed")


handles, labels = ax4.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right', ncol=5, fancybox=True, shadow=True, title="Nº of layers")
ax2.legend(loc='center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=True, title="Nº of hidden layers")
ax1.set_ylabel('CQV')
ax3.set_ylabel('CQV')
ax3.set_xlabel('Neurons per layer')
ax4.set_xlabel('Neurons per layer')
# plt.title('CQV of RVFL (continous) and ELM (dashed) models', loc='left')
# ax1.set_ylim(2,13)

# Draw titles for each subplot
props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
ax1.text(4.25, 0.50, 'Madrid', fontsize=18, verticalalignment='top', bbox=props)
ax2.text(4.18, 0.50, 'California', fontsize=18, verticalalignment='top', bbox=props)
ax3.text(4.18, 0.50, 'New york', fontsize=18, verticalalignment='top', bbox=props)
ax4.text(4.28, 0.50, 'Seattle', fontsize=18, verticalalignment='top', bbox=props)


plt.axis([0,5,0.00001,1])






















