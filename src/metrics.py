# src/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def calculate_gini(y_true, y_prob):
    """
    Calculates the Gini Coefficient.
    Relationship: Gini = 2 * AUC - 1
    """
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    return gini

def calculate_ks(y_true, y_prob):
    """
    Calculates the Kolmogorov-Smirnov (KS) statistic.
    Measures the maximum distance between the cumulative distribution 
    of 'Good' (target=0) and 'Bad' (target=1) customers.
    """
    df = pd.DataFrame({'target': y_true, 'prob': y_prob})
    
    df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
    
    df['cum_bads'] = df['target'].cumsum() / df['target'].sum()
    df['cum_goods'] = (1 - df['target']).cumsum() / (1 - df['target']).sum()
    
    ks_stat = (df['cum_bads'] - df['cum_goods']).abs().max()
    
    return ks_stat