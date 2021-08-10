import numpy as np
import os 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import auc, roc_curve
from sklearn.ensemble import RandomForestClassifier


def feat_select_permutation(X, y, n_perm=1000, alpha=95):

    """
    Function to select features based on feature importance 
    significantly larger than null (estimated through permutation)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features).
    y : array-like of shape (n_samples).
    n_perm : int, optional
    alpha : float, optional

    Returns
    -------
    feat_sel: bool array (n_features, )
        Boolean array for selecting features
    feat_impo (n_features, )
        Estimated feature importance
    sel_level (n_features, )
        Selection level for feature importance

    """    

    # Initialize variables
    rfc = RandomForestClassifier() 
    
    # Get feature importance
    rfc = rfc.fit(X, y)
    feat_impo = rfc.feature_importances_
    
    # Permute labels and get feature importance
    perm_feat_impo = []
    for n in range(n_perm):
        rgn = np.random.default_rng(n)
        y_perm = rgn.permutation(y)
        rfc = rfc.fit(X, y_perm)
        perm_feat_impo += [rfc.feature_importances_]
    
    # Compute limit of 95% confidence interval of null for each feature
    sel_level = np.percentile(np.vstack(perm_feat_impo), alpha, axis=0)
    
    # Select features which are significantly larger than null
    feat_sel = np.greater(feat_impo, sel_level)
    
    return feat_sel, feat_impo, sel_level


if __name__ == '__main__':
    
    # Run test
    path = r'C:\Users\ellah\OneDrive - Danmarks Tekniske Universitet\Project with NRU\scripts'
    os.chdir(path)    
    from np1_02_wrangling import X, y, attributeNames, M
    attributeNames = np.array(attributeNames)
    feat_sel, feat_impo, sel_level = feat_select_permutation(X, y, n_perm=10)
    print(attributeNames[feat_sel])
    
    df_data = np.vstack([attributeNames, feat_impo, sel_level]).T
    df = pd.DataFrame(df_data, columns=['feature', 'feat_impo', 'sel_level'])
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.sort_values(by='feat_impo', ascending=False)
    
    fig, ax = plt.subplots(figsize=(28,20))
    sns.barplot(x='feature', y='feat_impo', data=df, ax=ax, color='b')
    sns.lineplot(x='feature', y='sel_level', data=df, ax=ax, color='black')
    ax.set(xlabel='Feature importance', ylabel='Feature')
    plt.xticks(rotation=30, ha='right')
    # plt.legend(fontsize=20)
    plt.show()   
    

    # Initialize plot
    fig, ax = plt.subplots(figsize=(14,10))
    
    # Initialize variables
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    repititions = 5 # number of repeated CVs
    K = 5 # number of folds in outer CV loop
    K1 = 10 # number of folds in inner CV loop

    rfc = RandomForestClassifier() 
    
    for i in range(repititions):
        CV = StratifiedKFold(n_splits=K,shuffle=True)
        pred = 2*np.ones(len(y))
        # Perform two level cross validation

        # Outer CV loop
        for par, test in CV.split(X,y):
            CV1 = KFold(n_splits=K1,shuffle=True)
    
            # Initialize variables for inner loop
            feat_sel_array = np.empty((M,K1)) # for storing the feature importance
            aucs_inner = []
            pred_inner = 2*np.ones(len(y[par]))
            
            j = 0
            
            # Inner CV loop
            for train, val in CV1.split(X[par]):
                # Fit classifier and get selected features
                feat_sel, feat_impo, sel_level = feat_select_permutation(X[train], y[train], n_perm=10)
                feat_sel_array[:,j] = feat_sel
                
                # Fit classifier with selected features and get auc of fit
                X_train_selected = X[:,feat_sel]
                rfc = rfc.fit(X_train_selected[train],y[train])
                
                pred_inner[val] = rfc.predict_proba(X_train_selected[val])[:, 1]
                
                # Compute AUC
                fpr, tpr, thresholds = roc_curve(y[val], pred_inner[val])
                roc_auc = auc(fpr, tpr)
                aucs_inner.append(roc_auc)
            
                j+=1
        
            # Get index for highest auc
            max_auc = max(aucs_inner)
            max_index = aucs_inner.index(max_auc)
        
            # Get feature importance for the fold with the highest AUC
            feat_sel_outer = feat_sel_array[:,max_index].astype(bool)

            X_selected = X[:,feat_sel_outer]
            
            # Fit classifier
            rfc = rfc.fit(X_selected[par],y[par])

            pred[test] = rfc.predict_proba(X_selected[test])[:, 1]
        
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y, pred)
        roc_auc = auc(fpr, tpr)
        print ("Area under the ROC curve for fold {0}/{1}: {2}"
               .format(i+1, repititions, roc_auc))
        
        # Interpolate to get evenly spaced interval for tpr
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0]= 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        
    # Plot the "chance" curve
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    label='Chance', alpha=0.8)
        
    # Plot the mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' 
            % (mean_auc, std_auc),
            lw=2, alpha=0.8)
    
    # Plot the shaded standard deviation
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + 2 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 2 * std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                    label=r'$\pm$ 2 std. deviations')   
    
    # Plot layout
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title='Receiver Operating Characteristic',
           ylabel='True Positive Rate',
           xlabel='False Positive Rate')
    ax.legend(loc='lower right')
    
        
    plt.show()



