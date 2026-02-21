# src/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

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

def plot_roc_curve(y_true, scores, ax=None):
    """
    Plots the ROC curve and calculates AUC & Gini.
    """
    is_standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        is_standalone = True
        
    fpr, tpr, thresholds = roc_curve(y_true, -scores)
    roc_auc = auc(fpr, tpr)
    gini = (2 * roc_auc) - 1
    
    ax.plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})\nGini = {gini:.3f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate (Cumulative Good)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Cumulative Bad)', fontsize=12)
    ax.legend(loc='lower right', fontsize=12)
    
    if is_standalone:
        plt.tight_layout()
        plt.show()
        
    return roc_auc, gini


def plot_ks_curve(y_true, scores, ax=None):
    """
    Plots the Kolmogorov-Smirnov (KS) curve and finds the optimal cutoff.
    """
    is_standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        is_standalone = True
        
    df = pd.DataFrame({'target': y_true, 'score': scores})
    df = df.sort_values(by='score', ascending=True).reset_index(drop=True)
    
    df['good'] = (df['target'] == 0).astype(int)
    df['bad']  = (df['target'] == 1).astype(int)
    
    # Calculate cumulative proportions
    df['cum_good'] = df['good'].cumsum() / df['good'].sum()
    df['cum_bad']  = df['bad'].cumsum()  / df['bad'].sum()
    
    # Calculate KS Statistic
    df['ks_stat'] = np.abs(df['cum_bad'] - df['cum_good'])
    ks_max_idx = df['ks_stat'].idxmax()
    ks_stat = df['ks_stat'].max()
    ks_score = df.loc[ks_max_idx, 'score']
    
    ax.plot(df['score'], df['cum_bad'], color='#e74c3c', linewidth=2, label='Cumulative Bad (Default)')
    ax.plot(df['score'], df['cum_good'], color='#2ecc71', linewidth=2, label='Cumulative Good (Paid)')
    
    # Plot KS Maximum Distance Line
    ax.plot([ks_score, ks_score], 
            [df.loc[ks_max_idx, 'cum_good'], df.loc[ks_max_idx, 'cum_bad']], 
            color='black', linestyle='--', linewidth=2, 
            label=f'MAX KS = {ks_stat:.3f}\nat Score = {ks_score}')
    
    ax.set_title('Kolmogorov-Smirnov (KS) Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Credit Score', fontsize=12)
    ax.set_ylabel('Cumulative Proportion', fontsize=12)
    ax.legend(loc='lower right', fontsize=12)
    
    if is_standalone:
        plt.tight_layout()
        plt.show()
        
    return ks_stat, ks_score

def create_decile_table(y_true, scores):
    """
    Creates a Decile Analysis table for credit scoring.
    Sorts customers from highest risk (lowest score) to lowest risk (highest score)
    and calculates Bad Rate, Cumulative Bad, KS, and Lift for each decile.
    """
    df = pd.DataFrame({'target': y_true, 'score': scores})
    
    df = df.sort_values('score', ascending=True)
    
    # Divide into 10 equal deciles
    df['Decile'] = pd.qcut(df['score'].rank(method='first'), 10, labels=range(1, 11))
    
    overall_bad_rate = df['target'].mean()
    total_bad_population = df['target'].sum()
    total_good_population = (df['target'] == 0).sum()
    
    agg_df = df.groupby('Decile').agg(
        Min_Score=('score', 'min'),
        Max_Score=('score', 'max'),
        Total_Customers=('target', 'count'),
        Bad_Customers=('target', 'sum')
    ).reset_index()
    
    agg_df['Good_Customers'] = agg_df['Total_Customers'] - agg_df['Bad_Customers']
    agg_df['Bad_Rate_%'] = (agg_df['Bad_Customers'] / agg_df['Total_Customers']) * 100
    
    agg_df['Cum_Bad'] = agg_df['Bad_Customers'].cumsum()
    agg_df['Cum_Good'] = agg_df['Good_Customers'].cumsum()
    agg_df['Cum_Total_Customers'] = agg_df['Total_Customers'].cumsum()
    
    agg_df['%_Cum_Bad'] = (agg_df['Cum_Bad'] / total_bad_population) * 100
    agg_df['%_Cum_Good'] = (agg_df['Cum_Good'] / total_good_population) * 100
    
    agg_df['KS_Stat'] = np.abs(agg_df['%_Cum_Bad'] - agg_df['%_Cum_Good'])
    
    # Lift Calculations
    agg_df['Decile_Lift'] = (agg_df['Bad_Customers'] / agg_df['Total_Customers']) / overall_bad_rate
    
    agg_df['Cum_Lift'] = (agg_df['Cum_Bad'] / agg_df['Cum_Total_Customers']) / overall_bad_rate
    
    return agg_df

def plot_lift_gain_curves(decile_table):
    """
    Plots Cumulative Gain and Cumulative Lift charts based on the decile table.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- 1. CUMULATIVE GAINS CHART ---
    pop_perc = np.arange(10, 110, 10)
    
    axes[0].plot(pop_perc, decile_table['%_Cum_Bad'], marker='o', color='#e74c3c', linewidth=2, label='Model (Scorecard)')
    axes[0].plot([0, 100], [0, 100], linestyle='--', color='gray', label='Random Selection')
    
    axes[0].set_title('Cumulative Gains Chart', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('% of Population (Ranked by Risk)', fontsize=12)
    axes[0].set_ylabel('% of Total Bad Customers Captured', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)

    # --- 2. CUMULATIVE LIFT CHART ---
    axes[1].plot(pop_perc, decile_table['Cum_Lift'], marker='o', color='#3498db', linewidth=2, label='Cumulative Lift')
    axes[1].axhline(y=1, color='gray', linestyle='--', label='Baseline (Lift=1)')
    
    axes[1].set_title('Cumulative Lift Chart', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('% of Population (Ranked by Risk)', fontsize=12)
    axes[1].set_ylabel('Lift Multiplier (x)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_single_cutoff_metrics(y_true, scores, threshold):
    """
    Calculates key banking metrics for a specific score threshold.
    """
    approved_mask = scores >= threshold
    total_customers = len(y_true)
    num_approved = approved_mask.sum()
    
    bads_in_approved = y_true[approved_mask].sum()
    
    approval_rate = (num_approved / total_customers) * 100
    portfolio_bad_rate = (bads_in_approved / num_approved) * 100 if num_approved > 0 else 0
    
    # Risk Mitigation Power (How much of total bads did we reject?)
    total_bads = y_true.sum()
    bads_rejected = total_bads - bads_in_approved
    bad_capture_rate = (bads_rejected / total_bads) * 100
    
    return {
        'Cutoff': threshold,
        'Approval_Rate_%': approval_rate,
        'Portfolio_Bad_Rate_%': portfolio_bad_rate,
        'Bad_Capture_Rate_%': bad_capture_rate
    }

def simulate_cutoff_scenarios(y_true, scores, step=10):
    """
    Iterates through score ranges to create a simulation matrix.
    """
    min_score = int(scores.min() // 10 * 10)
    max_score = int(scores.max() // 10 * 10)
    
    scenarios = []
    for t in range(min_score, max_score + step, step):
        scenarios.append(calculate_single_cutoff_metrics(y_true, scores, t))
        
    return pd.DataFrame(scenarios)

def plot_tradeoff_curve(cutoff_matrix):
    """
    Visualizes Approval Rate vs Portfolio Bad Rate
    (Banking Risk Reporting Standard Style)
    """

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    approval_color = "#2E86C1"   # corporate blue
    risk_color = "#C0392B"       # risk red

    # --- Approval Rate ---
    line1, = ax1.plot(
        cutoff_matrix['Cutoff'],
        cutoff_matrix['Approval_Rate_%'],
        marker='o',
        linewidth=2.5,
        color=approval_color,
        label="Approval Rate (%)"
    )

    # --- Bad Rate ---
    line2, = ax2.plot(
        cutoff_matrix['Cutoff'],
        cutoff_matrix['Portfolio_Bad_Rate_%'],
        marker='s',
        linewidth=2.5,
        linestyle='--',
        color=risk_color,
        label="Portfolio Bad Rate (%)"
    )

    ax1.set_xlabel("Score Cut-off Threshold", fontsize=12, labelpad=10)

    ax1.set_ylabel(
        "Approval Rate (%)",
        fontsize=12,
        color=approval_color
    )

    ax2.set_ylabel(
        "Bad Rate in Approved (%)",
        fontsize=12,
        color=risk_color
    )

    ax1.tick_params(axis='y', colors=approval_color)
    ax2.tick_params(axis='y', colors=risk_color)

    plt.title(
        "Approval vs Risk Trade-off Curve",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    ax1.grid(True, linestyle='--', alpha=0.25)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]

    ax1.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    plt.tight_layout()

    plt.show()