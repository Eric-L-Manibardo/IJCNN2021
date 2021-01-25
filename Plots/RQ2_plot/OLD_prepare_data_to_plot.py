# -*- coding: utf-8 -*-

# The aim of this script is to produce CSV files
# of data to plot, by extracting the mean of 10 locations of a city

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set up the matplotlib figure
sns.despine(left=True)
sns.set(style="whitegrid")


neurons = ['1','10','50','100','500','1000'] # neurons per layer
hidden = ['2','4','6','8','10'] #hidden layers

df = pd.read_excel('CQV_Madrid.xlsx')
# extract one horizon only
h=0
df = df.iloc[:, h*10 :h*10 + 11]

# extract 3 30 first values for RVFL and 0 last values for ELM
rvfl = df.iloc[:30,: ]
elm = df.iloc[90:120,:]

# compute the mean per row
rvfl['CQV'] = df.mean(axis=1)
elm['CQV'] = df.mean(axis=1)

# add x labels to dataframe

n = list()
for i in range(len(hidden)):
    for j in range(len(neurons)):
        n.append(neurons[j])
rvfl['Neurons per layer']= n
elm['Neurons per layer']= n

palette_online = ["#FF595E", "#FFCA3A" , "#8AC926", "#1982C4",'#6A4C93']
# fig = plt.figure()
# plot multiple layer configurations linestyle="-"


    # ax.plot(elm.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], elm.iloc[6*i : 6*i + 6,: ]['CQV'],label = hidden[i],color=palette_online[i],linestyle="dashed")
    

fig, axs = plt.subplots(2, 2, sharex=True)
(ax1, ax2), (ax3, ax4) = axs
fig.suptitle('Sharing x per column, y per row')
for i in range(len(hidden)):
    ax1.semilogy(rvfl.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], rvfl.iloc[6*i : 6*i + 6,: ]['CQV'],label = hidden[i],color=palette_online[i])
    ax1.semilogy(elm.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], elm.iloc[6*i : 6*i + 6,: ]['CQV'],label = hidden[i],color=palette_online[i],linestyle="dashed")

for i in range(len(hidden)):
    ax2.semilogy(rvfl.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], rvfl.iloc[6*i : 6*i + 6,: ]['CQV'],label = hidden[i],color=palette_online[i])
    
    
for i in range(len(hidden)):
    ax3.semilogy(rvfl.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], rvfl.iloc[6*i : 6*i + 6,: ]['CQV'],label = hidden[i],color=palette_online[i])
    
for i in range(len(hidden)):
    ax4.semilogy(rvfl.iloc[6*i : 6*i + 6,: ]['Neurons per layer'], rvfl.iloc[6*i : 6*i + 6,: ]['CQV'],label = hidden[i],color=palette_online[i])


handles, labels = ax4.get_legend_handles_labels()
fig.legend(handles, labels, loc='center', ncol=5, fancybox=True, shadow=True)






