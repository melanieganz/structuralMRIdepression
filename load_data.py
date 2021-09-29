#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Clean names
clean_names = {
    'accumbens-area': 'Accumbens area',
    'amygdala': 'Amygdala',
    'bankssts': 'Banks of the STS',
    'caudalanteriorcingulate': 'Caudal anterior cingulate',
    'caudalmiddlefrontal': 'Caudal middle frontal',
    'caudate': 'Caudate',
    'cerebellum-cortex': 'Cerebellum cortex',
    'hippocampus': 'Hippocampus',
    'cuneus': 'Cuneus',
    'entorhinal': 'Entorhinal',       
    'frontalpole': 'Frontal pole',
    'fusiform': 'Fusiform',
    'inferiorparietal': 'Inferior parietal',
    'inferiortemporal': 'Inferior temporal',
    'insula': 'Insula',
    'ICV': 'Brain volume',
    'isthmuscingulate': 'Isthmus of cingulate',
    'lateraloccipital': 'Lateral occipital',
    'lateralorbitofrontal': 'Lateral orbitofrontal',
    'lateral-ventricle': 'Lateral ventricle',
    'lingual': 'Lingual',
    'mean_thick': 'Mean Thickness',
    'medialorbitofrontal': 'Medial orbitofrontal',
    'middletemporal': 'Middle temporal',
    'pallidum': 'Pallidum',
    'paracentral': 'Paracentral',
    'parahippocampal': 'Parahippocampal',
    'parsopercularis': 'Pars opercularis', 
    'parsorbitalis': 'Pars orbitalis',
    'parstriangularis': 'Pars triangularis',
    'pericalcarine': 'Pericalcarine',
    'postcentral': 'Postcentral',
    'posteriorcingulate': 'Posterior cingulate',
    'precentral': 'Precentral',
    'precuneus': 'Precuneus',
    'putamen': 'Putamen',
    'rostralanteriorcingulate': 'Rostral anterior cingulate',
    'rostralmiddlefrontal': 'Rostral middle frontal',
    'superiorfrontal': 'Superior frontal',
    'superiorparietal': 'Superior parietal',
    'superiortemporal': 'Superior temporal',
    'supramarginal': 'Supramarginal',
    'temporalpole': 'Temporal pole',
    'thalamus': 'Thalamus',
    'transversetemporal': 'Tranverse temporal'
}

def load_data(
            csv_file='raw_data.csv',
            out_type='mean',
            out_csv=None,
            drop_var=None,
            standardize=False,
        ):
    
    # Load raw data
    df = pd.read_csv(csv_file)
    
    # Remove participants with no hamd_change_week8 values
    df = df[np.logical_not(pd.isna(df['hamd_change_week8']))]
    
    # Remove participants younger than 21
    df = df[df.age > 21]
    
    # Remove healthy controls
    df = df[df['Person status'] != 'Healthy Control']
    
    # Output data.frame
    X = pd.DataFrame()
    
    # Age
    X['age'] = df['age']
    
    # Binary encode sex and single_recurrent
    X['sex'] = df['sex'].map({'Male' : 0, 'Female' : 1})
    
    # Single/recurrent MDD
    X['single_recurrent'] = df['single_recurrent'].map({'Single' : 0,
                            'Recurrent' : 1})
    
    # baseline HAMD
    X['hamd_base'] = df['hamd_base']
        
    regions = df.columns[[col.startswith('lh') for col in df.columns]]
    regions = [region.replace('lh.', '') for region in regions]  
        
    if out_type == 'mean':     
        for region in regions:
            X['mean.' + region] = np.mean([df['lh.' + region], df['rh.' + region]], axis=0)

    if out_type == 'lr':
        for region in regions:
            X['lh.' + region] = df['lh.' + region]
        for region in regions:
            X['rh.' + region] = df['rh.' + region]
        
    if out_type == 'min.max':
        for region in regions:
            X['min.' + region] = np.min([df['lh.' + region], df['rh.' + region]], axis=0)
        for region in regions:
            X['max.' + region] = np.max([df['lh.' + region], df['rh.' + region]], axis=0)

    # Add ICV
    X['ICV'] = np.float64(df['ICV'])  # somehow, casting is necessary to avoid NaN (???)

    # Binary (threshold) encode the "change" column so
    # that 0 equals a change less than 50% and 1 equals a change larger than 50%
    threshold = -50
    y = df['hamd_change_week8'].gt(threshold) # .astype(int)
    y = y.map({True : 0, False : 1})
            
    if out_csv is not None:
        df = pd.concat([X, y], axis=1)
        df.columns = list(X.columns) + ['group']
        df.to_csv(out_csv, index=False)
        
    if drop_var is not None:
        X = X.drop(drop_var, axis=1)
        
    if standardize:
        X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        
    # Assign feature names
    feature_names = np.array(X.columns)
    
    return X.to_numpy(), y.to_numpy(), feature_names