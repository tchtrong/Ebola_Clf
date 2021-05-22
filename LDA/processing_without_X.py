#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 00:55:55 2021

@author: binn
"""

#%%
from sklearn.decomposition import LatentDirichletAllocation as lda
import pandas as pd

#%%
#Load dataset and labels
dataset = pd.read_csv("../csv/freq_matrix_without_X.csv")

#%%
#Using LDA with 6 topics corresponding to 6 species
lda_6tp = lda(n_components=6, random_state=(0), n_jobs=-1)
lda_6tp.fit(dataset)
pd.DataFrame(lda_6tp.transform(dataset)).to_csv("csv/lda_6tp_without_X.csv")

#%%
#
for i in range(10, 310, 10):
    lda_i_tp = lda(n_components=i, random_state=(0), n_jobs=-1)
    lda_i_tp.fit(dataset)
    pd.DataFrame(lda_i_tp.transform(dataset)).to_csv("csv/lda_{}tp_without_X.csv".format(i))
    
    
    
    
    