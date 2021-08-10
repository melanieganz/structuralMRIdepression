# -*- coding: utf-8 -*-
"""
Created on Sat May 15 10:06:29 2021

@author: ellah
"""

#%% Import packages

import numpy as np
import pandas as pd
import os

#%% Load data

# Set data dir
data_dir = r'C:\Users\ellah\OneDrive - Danmarks Tekniske Universitet\Project with NRU\data'

# Load data
dd = pd.read_csv(os.path.join(data_dir, 'np1_mri_mean_data.csv')) 
# change file name to get different features for regions:
# min/max, left/right or mean of regions

#%% Convert data frame to X, y format

# Remove id from data frame
dd.drop(["id"], axis = 1, inplace = True)

# Get attribute names
attributeNames = dd.columns.values.tolist() 

# convert data frame to numpy array
X = dd.to_numpy()

# Split dataset into features and target vector
change_idx = attributeNames.index('change')
y = X[:, change_idx]

X_cols = list(range(0,change_idx)) + list(range(change_idx+1,len(attributeNames)))
X = X[:,X_cols]

# Remove "change" from attribute list
attributeNames.remove('change')

# Return the number of rows and number of columns (attributes)
N, M = X.shape

# Standardize data
mu = np.mean(X, 0)
sigma = np.std(X, 0)

X = (X - mu) / sigma

# Get class names
C = 2
classNames = ["Non-responder", "Responder"]