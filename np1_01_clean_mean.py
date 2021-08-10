# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:34:32 2021

@author: ellah
"""

#%% Import packages

import pandas as pd
import os

#%% Read meta data 

# Set data dir
data_dir = r'C:\Users\ellah\OneDrive - Danmarks Tekniske Universitet\Project with NRU\data'

# Load meta data
dd_meta = pd.read_csv(os.path.join(data_dir, 'MR_NP1_HC_DBproject_baseline.csv'))
dd_meta = dd_meta[["SubjID", "Gender", " Age at MR ", 
                   "RH-MR Lognumber", "HAMD-6 score - Baseline",
                   "Single or recurrent MDD episode?"]]
# Rename columns
dd_meta.rename(columns = {'SubjID' : 'id',
                          'Gender' : 'sex',
                          ' Age at MR ' : 'age',
                          'RH-MR Lognumber' : 'lognumber',
                          'HAMD-6 score - Baseline' : 'hamd_base',
                          'Single or recurrent MDD episode?' : 'single_recurrent'},
               inplace = True)

# Load outcome data
dd_outcome = pd.read_csv(os.path.join(data_dir, 'MR_NP1_HC_DBproject_outcome.csv'))
dd_outcome = dd_outcome[["RH-MR Lognumber", 
         "NP1 secondary outcome - Percent change in HAMD-6 at week 8 compared to baseline"]]
# Rename columns
dd_outcome.columns = ['lognumber', 'change']

# Add outcome to meta dd
dd_meta = pd.merge(dd_meta, dd_outcome, on = "lognumber")
dd_meta.drop(["lognumber"], axis = 1, inplace = True)

#%% Read cortical thickness data

# Right hemisphere
dd_rh = pd.read_csv(os.path.join(data_dir, 'NP1_v71_aparc_stats_RH.txt'), delimiter ="\t")
dd_rh.rename(columns = {'rh.aparc.thickness' : 'id'}, inplace = True)
dd_rh.drop(["BrainSegVolNotVent", "eTIV"], axis=1, inplace= True)
dd_rh.columns = dd_rh.columns.str.rstrip('_thickness')

# Left hemisphere
dd_lh = pd.read_csv(os.path.join(data_dir, 'NP1_v71_aparc_stats_LH.txt'), delimiter ="\t")
dd_lh.rename(columns = {'lh.aparc.thickness' : 'id'}, inplace = True)
dd_lh.drop(["BrainSegVolNotVent", "eTIV"], axis=1, inplace= True)
dd_lh.columns = dd_lh.columns.str.rstrip('_thickness')

# Merge the dataframes
dd_cortical = pd.merge(dd_rh, dd_lh, on = "id")

#%% Calculate the mean of each region

# Create list with region names
region_names = list(dd_rh.columns[1:])
region_names = [str(i).replace( 'rh_', '') for i in region_names]

# Create data frame for storing region means
dd_cortical_mean = dd_lh.iloc[:,0].to_frame()

# Add the mean of each region to dd_mean
for i in range(1,len(dd_lh.columns)):
    j = i - 1 
    left = dd_lh.iloc[:,i]
    right = dd_rh.iloc[:,i]
    dd_tmp = pd.concat([left,right], axis = 1)
    dd_cortical_mean[region_names[j]] = dd_tmp.mean(axis=1)
    
#%% Read subcortical data

dd_subcortical = pd.read_csv(os.path.join(data_dir, 'LandRvolumes.csv'))
dd_subcortical.rename(columns = {'SubjID' : 'id'}, inplace = True)

# Create list with region names
sub_region_names = list(dd_subcortical[dd_subcortical.columns[1::2]].columns[:-1].str[1:])

# Create data frame for storing region means
dd_subcortical_mean = dd_subcortical.iloc[:,0].to_frame()

# Add the mean of each region to dd_mean
for i in range(len(sub_region_names)):
    k = i+(i+1)
    j = k+1 
    left = dd_subcortical.iloc[:,k]
    right = dd_subcortical.iloc[:,j]
    dd_tmp = pd.concat([left,right], axis = 1)
    dd_subcortical_mean[sub_region_names[i]] = dd_tmp.mean(axis=1)

#%% Join data frames

dd_mri = pd.merge(dd_subcortical_mean, dd_cortical_mean, on = 'id')
dd = pd.merge(dd_meta, dd_mri, on = "id")

#%% Data clean-up 

# Keep only the participants who has a value in the "change" column
dd = dd.dropna()

# Remove participants younger than 21
dd = dd[dd.age > 21]

# Binary encode sex and single_recurrent
dd['sex'] = dd['sex'].map({'Male' : 0,
                           'Female' : 1})
dd['single_recurrent'] = dd['single_recurrent'].map({'Single' : 0,
                           'Recurrent' : 1})

# Binary (threshold) encode the "change" column so
# that 0 equals a change less than 50% and 1 equals a change larger than 50%
threshold = -50
dd['change'] = dd['change'].gt(threshold) # .astype(int)
dd['change'] = dd['change'].map({True : 0,
                           False : 1})
class_balance = dd['change'].value_counts()

# Save analysis ready data frame
dd.to_csv(os.path.join(data_dir, 'np1_mri__mean_data.csv'),
          index = False)


dd['sex'].count()
