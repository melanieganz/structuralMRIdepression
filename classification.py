#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from multiprocessing import Pool
from itertools import product    
from load_data import load_data, clean_names


def run_classification(X, y, seed, permute=False):
   
    """ Wrapper for the parallel processing of classification """
    
    if permute:
        rgn = np.random.default_rng(seed)
        y = rgn.permutation(y.copy())
    
    # Initialize
    K = 10  # number of folds in CV loop
    rfc = RandomForestClassifier(
            class_weight='balanced',
            random_state=seed           
        )
    CV = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    
    # CV loop
    proba = np.zeros(len(y))
    pred = np.zeros(len(y))
    feat_impo = []
    for train, test in CV.split(X, y):
                   
        # Fit classifier
        rfc = rfc.fit(X[train, :], y[train])
        feat_impo.append(rfc.feature_importances_)
        
        # Get predictions
        proba[test] = rfc.predict_proba(X[test, :])[:, 1]
        pred[test] = rfc.predict(X[test, :])
    
    # Compute ROC curve and AUROC
    fpr, tpr, thresholds = roc_curve(y, proba)
    auroc = auc(fpr, tpr)
    
    # Compute accuracy, sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    acc = (tp + tn) / len(y)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    
    # Interpolate to get evenly spaced interval for tpr
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0]= 0.0
    interp_tpr[-1]= 1.0
    
    return auroc, acc, sens, spec, interp_tpr, feat_impo


#%% Perform classification

out_type = 'mean'

""" Load data """

X, y, feature_names = load_data(
        out_type=out_type,
        out_csv='data_' + out_type + '.csv',
        drop_var=['age', 'sex', 'hamd_base', 'single_recurrent']
    )

# Initialize variables
mean_fpr = np.linspace(0, 1, 1000)  # for interpolating ROC
n_rep = 1000  # number of repeated CVs
   
""" Run classification """
   
# Run CV loops in parallel
params = product([X], [y], range(n_rep))
with Pool(processes=20) as pool:
    out = pool.starmap(run_classification, params)
aurocs, acc, sens, spec, tprs, feat_impo = zip(*out)

# Run CV loops in parallel - permuted y
params = product([X], [y], range(n_rep), [True])
with Pool(processes=20) as pool:
    out = pool.starmap(run_classification, params)
perm_aurocs, perm_acc, perm_sens, perm_spec, perm_tprs, perm_feat_impo = zip(*out)

# Run CV loops in parallel - with one additional random variable
rng = np.random.RandomState(seed=42)
X_ = [np.hstack((X, rng.random([X.shape[0], 1]))) for _ in range(n_rep)]
params = [(X_, y, seed) for X_, y, seed in zip(
          [np.hstack((X, rng.random([X.shape[0], 1]))) for _ in range(n_rep)],
          [y for _ in range(n_rep)],
          range(n_rep)
    )]
with Pool(processes=20) as pool:
    out = pool.starmap(run_classification, params)
rand_aurocs, rand_acc, rand_sens, rand_spec, rand_tprs, rand_feat_impo = zip(*out)  

#%% Compare AUROCS from normal vs. permuted and print out other performance metrics

plt.hist(np.vstack((aurocs, perm_aurocs)).T, bins=50)
print(ttest_ind(aurocs, perm_aurocs, equal_var=False))
print('Mean AUROC: %f' % np.mean(aurocs))
print('Mean accuracy: %f' % np.mean(acc))
print('Mean sensitivity: %f' % np.mean(sens))
print('Mean specificity: %f' % np.mean(spec))

#%%
""" Plot mean """

fig, ax = plt.subplots(figsize=(8, 8))

# Plot mean roc
mean_tpr = np.mean(np.vstack(tprs), axis=0)
mean_auc = np.mean(aurocs)
std_auc = np.std(aurocs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC, AUC (mean, SD): %0.2f $\pm$ %0.2f)'
        % (mean_auc, std_auc),
        lw=2, alpha=0.8)

# Plot mean permuted
mean_perm_tpr = np.mean(np.vstack(perm_tprs), axis=0)
mean_perm_auc = np.mean(perm_aurocs)
std_perm_auc = np.std(perm_aurocs)
ax.plot(mean_fpr, mean_perm_tpr, 'g--',
        label=r'Mean ROC - permuted data, AUC (mean, SD): %0.2f $\pm$ %0.2f)'
        % (mean_perm_auc, std_perm_auc),
        lw=2, alpha=0.8)

# Compute standard deviation
sd_tpr = np.std(np.vstack(tprs), axis=0)
tprs_lower = mean_tpr - sd_tpr
tprs_upper = mean_tpr + sd_tpr
    
# Compute permuted standard deviation
sd_perm_tpr = np.std(np.vstack(perm_tprs), axis=0)
perm_tprs_lower = mean_perm_tpr - sd_perm_tpr
perm_tprs_upper = mean_perm_tpr + sd_perm_tpr

# Plot standard deviations and identity    
ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                color='b', alpha=0.2)          
ax.fill_between(mean_fpr, perm_tprs_lower, perm_tprs_upper,
                color='g', alpha=0.2)    
ax.plot(mean_fpr, mean_fpr, 'r--', label='Identity')

# Plot layout
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)
ax.legend(loc='lower right')    
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
    
plt.savefig('roc_' + out_type + '.svg', dpi=300)
plt.savefig('roc_' + out_type + '.png', dpi=300)

#%%
""" Plot feature importance """

# Clean up feature names if needed
features = feature_names.copy()
if out_type == 'lr':
    ind = np.logical_or(
            [feat.startswith('lh.') for feat in features],
            [feat.startswith('rh.') for feat in features]
        )  
    prefix = np.array([feat[0:2] for feat in features])
    prefix = ['Left ' if p == 'lh' else
              'Right ' if p == 'rh' else
              '' for p in prefix]       
    features[ind] = [feat[2:] for feat in features[ind]]
elif out_type == 'min.max':
    ind = np.logical_or(
            [feat.startswith('min.') for feat in features],
            [feat.startswith('max.') for feat in features]
        )  
    prefix = [feat[0:3] for feat in features[ind]]
    prefix = ['Min ' if p == 'min' else
              'Max ' if p == 'max' else
              '' for p in prefix]  
    features[ind] = [feat[3:] for feat in features[ind]]
elif out_type == 'mean':
    prefix = ['']*len(features)
    ind = [feat.startswith('mean') for feat in features] 
    features[ind] = [feat[4:] for feat in features[ind]]
features = [feat.replace('.cort.', '') for feat in features]
features = [feat.replace('.subcort.', '') for feat in features]        
features = np.array([clean_names[feat] for feat in features])
features[ind] = [p + f for p, f in zip(prefix, features[ind])] 

# Compute random mean and 95% confidence interval
mean_feat_impo = np.mean(np.vstack(feat_impo), axis=0)
sd_feat_impo = np.std(np.vstack(feat_impo), axis=0)
feat_impo_lower = mean_feat_impo - sd_feat_impo
feat_impo_upper = mean_feat_impo + sd_feat_impo   

# Compute random mean and 95% confidence interval
mean_rand_feat_impo = np.ones(len(features))*np.mean(np.vstack(rand_feat_impo)[:, -1])
sd_rand_feat_impo = np.ones(len(features))*np.std(np.vstack(rand_feat_impo)[:, -1])
rand_feat_impo_lower = mean_rand_feat_impo - sd_rand_feat_impo
rand_feat_impo_upper = mean_rand_feat_impo + sd_rand_feat_impo


# Create figure
fig, ax = plt.subplots(figsize=(14, 6))

# Get ordering
ind = np.argsort(np.mean(np.vstack(feat_impo), axis=0))[::-1]

# Normal
ax.plot(features[ind], mean_feat_impo[ind], label='Normal processing', color='b')
ax.fill_between(features[ind], feat_impo_lower[ind], feat_impo_upper[ind],
                alpha=0.2, color='b')


# Random
ax.plot(features[ind], mean_rand_feat_impo, label='Random', color='orange')
ax.fill_between(features[ind], rand_feat_impo_lower, rand_feat_impo_upper,
                alpha=0.2, color='orange')

ax.set_xlabel(None)
ax.set_ylabel('Feature Importance', fontsize=16)
ax.legend().set_title('')
ax.tick_params(axis='x', rotation=90)
plt.tight_layout()    

plt.savefig('feat_impo_' + out_type + '.svg', dpi=300)
plt.savefig('feat_impo_' + out_type + '.png', dpi=300)

#%% Plot dendogram and correlation matrix

features = feature_names.copy()
features = [feat.replace('mean.cort.', '') for feat in features]
features = [feat.replace('mean.subcort.', '') for feat in features]        
features = np.array([clean_names[feat] for feat in features])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
stats = spearmanr(X)
corr = stats.correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(
    corr_linkage, labels=features, ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro['ivl']))

pos = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
fig.colorbar(pos, ax=ax2)
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()

plt.savefig('dendogram_corr.png', dpi=300)
plt.savefig('dendogram_corr.svg', dpi=300)
plt.show() 
