#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:45:09 2021

@author: eric
"""

# =============================================================================
# PLOTTER CD PLOT 
# =============================================================================

# Libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

np.random.seed = 0

# Input

N = 9 # Models
M = 40 # Datasets


nameModels = ['LV','sRVFL', 'RVFL', 'dRVFL', 'edRVFL', 'sELM','ELM', 'ETR', 'MLP']

# results1 = np.random.rand(M,N) #idealmente, leer de un fichero
df = pd.read_csv('overall.csv') # leo fichero
df = df.iloc[:,1:] # selecciono solo datos


#plot heatmap
# y_axis_labels = ['Madrid \n h=1','Madrid \n h=2','Madrid \n h=3','Madrid \n h=4',
#                  'California \n h=1','California \n h=2','California \n h=3','California \n h=4',
#                  'New York \n h=1','New York \n h=2','New York \n h=3','New York \n h=4',
#                  'Seattle \n h=1','Seattle \n h=2','Seattle \n h=3','Seattle \n h=4'] 
y_axis_labels = ['h=1','h=2','h=3','h=4',
                 'h=1','h=2','h=3','h=4',
                 'h=1','h=2','h=3','h=4',
                 'h=1','h=2','h=3','h=4']

sns.set(font_scale=1.0)
# cmap='YlGnBu'
ax = sns.heatmap(df,vmin=0.6, vmax=1, annot=True,cmap='Reds',
                 yticklabels=y_axis_labels,cbar=False,
                 square=True,fmt='.3f')

ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

# split axes of heatmap to put colorbar
ax_divider = make_axes_locatable(ax)
# define size and padding of axes for colorbar
cax = ax_divider.append_axes('top', size = '5%', pad = '2%')
# make colorbar for heatmap. 
# Heatmap returns an axes obj but you need to get a mappable obj (get_children)
colorbar(ax.get_children()[0], cax = cax, orientation = 'horizontal')
# locate colorbar ticks
cax.xaxis.set_ticks_position('top')
ax.yaxis.tick_right()
ax.yticks(rotation=0) 


bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)




plt.show()