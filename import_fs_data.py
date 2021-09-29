#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from os.path import join

recon_dir = '/mrdata/np1/FSrecon/proc/MR/recon_v71_edit'

# Load metadata
df = pd.read_csv('MR_NP1_HC_DBproject_baseline.csv')
df.rename(
    columns = {'SubjID' : 'id',
                'Gender' : 'sex',
                ' Age at MR ' : 'age',
                'RH-MR Lognumber' : 'subjects',
                'HAMD-6 score - Baseline' : 'hamd_base',
                'Single or recurrent MDD episode?' : 'single_recurrent'},
    inplace=True)

# Load outcome data
dd_outcome = pd.read_csv('MR_NP1_HC_DBproject_outcome.csv')
dd_outcome = dd_outcome[["RH-MR Lognumber", 
         "NP1 secondary outcome - Percent change in HAMD-6 at week 8 compared to baseline"]]
# Rename columns
dd_outcome.columns = ['subjects', 'hamd_change_week8']

# Add outcome to meta dd
df = pd.merge(df, dd_outcome, on = 'subjects')

aparc_col_names = 'StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd'.split(' ')
aseg_col_names = 'Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange'.split(' ')

subcort_regions = [
        'Lateral-Ventricle',
        'Cerebellum-Cortex',
        'Thalamus',
        'Caudate',
        'Putamen',
        'Pallidum',
        'Hippocampus',
        'Amygdala',
        'Accumbens-area'
    ]

# Append thicknes data
for hemi in ['lh', 'rh']:
    
    columns = None
    thick = []
    
    for subject in df.subjects:
    
        # Load stat file
        stat_file = join(recon_dir, subject + '_GD', 'stats', hemi + '.aparc.stats')
        with open(stat_file) as f:
            lines = np.array(f.readlines())
       
        # Extract mean thickness
        mean_thick = float(lines[[l.startswith('# Measure Cortex, MeanThickness')
                                  for l in lines]][0].split(' ')[-2][:-1])
        
        # Clean up stats
        lines = list(filter(lambda x: not x.startswith('#'), lines))
        lines = [list(filter(lambda x: x != '', l.strip().split(' '))) for l in lines]
        df_ = pd.DataFrame(np.vstack(lines), columns=aparc_col_names)
        
        # Make sur we have column headers
        if columns is None:
            columns = df_['StructName']
                
        # Store data
        thick += [list(df_['ThickAvg']) + [mean_thick]]
    
    df_ = pd.DataFrame(
            np.vstack(thick),
            columns=['.'.join([hemi, 'cort', region])
                     for region in list(columns) + ['mean_thick']],
            dtype=float
         )
    df = pd.concat([df, df_], axis=1)

# Append volume data
columns = None
volumes = []
for subject in df.subjects:

     # Load stat file
    stat_file = join(recon_dir, subject + '_GD', 'stats', 'aseg.stats')
    with open(stat_file) as f:
        lines = np.array(f.readlines())
    
    # Extract mean thickness
    brain_vol = float(lines[[l.startswith('# Measure BrainSegNotVent')
                          for l in lines]][0].split(' ')[-2][:-1])
    
    # Clean up stats
    lines = list(filter(lambda x: not x.startswith('#'), lines))
    lines = [list(filter(lambda x: x != '', l.strip().split(' '))) for l in lines]
    df_ = pd.DataFrame(np.vstack(lines), columns=aseg_col_names)
    keep_region = [region in ['Left-' + region_ for region_ in subcort_regions] or
                   region in ['Right-' + region_ for region_ in subcort_regions]
                   for region in df_['StructName']]
    df_ = df_.iloc[keep_region]

    if columns is None:
        columns = df_['StructName']
        columns = np.array([region.replace('Left-', 'lh.subcort.').replace('Right-', 'rh.subcort.').lower()
                   for region in columns])
            
    volumes += [list(df_['Volume_mm3']) + [brain_vol]] 

df_ = pd.DataFrame(np.vstack(volumes), columns=list(columns) + ['ICV'], dtype=float)
df = pd.concat([df, df_], axis=1)

# Save analysis ready data frame
df.to_csv('raw_data.csv', index = False)



