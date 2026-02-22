# src/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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

def calculate_psi(expected, actual, buckets=10):
    """
    Calculates the Population Stability Index (PSI).
    expected: Baseline distribution (Train scores)
    actual: New distribution (Test/Production scores)
    """
    def scale_range(data, bins):
        breakpoints = np.linspace(0, 100, bins + 1)
        breakpoints = np.percentile(data, breakpoints)
        breakpoints[0] = -np.inf 
        breakpoints[-1] = np.inf 
        return np.unique(breakpoints)

    breakpoints = scale_range(expected, buckets)
    
    expected_counts = pd.cut(expected, bins=breakpoints).value_counts(sort=False)
    actual_counts = pd.cut(actual, bins=breakpoints).value_counts(sort=False)
    
    # Normalize
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    expected_percents = expected_percents.replace(0, 0.0001)
    actual_percents = actual_percents.replace(0, 0.0001)
    
    # PSI Formula: (Actual% - Expected%) * ln(Actual% / Expected%)
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    total_psi = np.sum(psi_values)
    
    psi_df = pd.DataFrame({
        'Bucket': expected_counts.index,
        'Expected_%': expected_percents.values * 100,
        'Actual_%': actual_percents.values * 100,
        'PSI_Contribution': psi_values.values
    })
    
    return total_psi, psi_df

def calculate_csi(train_df, test_df, feature_list, buckets=10):
    """
    Calculates the Characteristic Stability Index (CSI) for multiple features.
    """
    csi_results = {}
    for feature in feature_list:
        csi_val, _ = calculate_psi(train_df[feature], test_df[feature], buckets)
        csi_results[feature] = csi_val
    
    return pd.Series(csi_results).sort_values(ascending=False)

def plot_psi_distribution(psi_details):
    """
    Plots the expected vs actual distribution from the PSI details table.
    """
    plt.figure(figsize=(12, 6))
    
    buckets = np.arange(len(psi_details))
    width = 0.35
    
    # Plot Bars
    plt.bar(buckets - width/2, psi_details['Expected_%'], width, label='Train (Expected)', color='#3498db', alpha=0.8)
    plt.bar(buckets + width/2, psi_details['Actual_%'], width, label='Test (Actual)', color='#e74c3c', alpha=0.8)
    
    plt.title('PSI Stability: Score Distribution Comparison', fontsize=15, fontweight='bold')
    plt.xlabel('Score Buckets (Low to High Risk)', fontsize=12)
    plt.ylabel('Population Percentage (%)', fontsize=12)
    plt.xticks(buckets, [f"B{i}" for i in range(len(psi_details))])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_csi_report(csi_series, use_log=True):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    csi_sorted = csi_series.sort_values(ascending=True)
    colors = ['#27ae60' if x < 0.1 else '#f1c40f' if x < 0.25 else '#e74c3c' for x in csi_sorted]
    
    ax.barh(csi_sorted.index, csi_sorted.values + 1e-7, color=colors, alpha=0.8)
    
    if use_log:
        ax.set_xscale('log')
        ax.set_xlabel('CSI Value (Logarithmic Scale)', fontsize=12, labelpad=10)
    else:
        ax.set_xlabel('CSI Value', fontsize=12, labelpad=10)

    ax.axvline(x=0.1, color='orange', linestyle='--', label='Warning (0.1)')
    ax.axvline(x=0.25, color='red', linestyle='--', label='Critical (0.25)')
    
    fig.suptitle('Characteristic Stability Index (CSI) - Log Scale View', 
                 fontsize=18, 
                 fontweight='bold', 
                 x=0.5) 
    
    ax.set_ylabel('Features', fontsize=10, fontweight='bold')
    
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(axis='x', which="both", alpha=0.3) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()

def plot_quantile_calibration_curve(y_true, y_prob, n_bins=10):
    """
    Plots a calibration curve using quantile-based binning to handle imbalanced data.
    Each bin contains the same number of samples.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='s', linewidth=2, label='Model (Quantile Bins)')
    plt.plot([0, prob_pred.max()], [0, prob_pred.max()], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability (Quantile-based PD)', fontsize=12)
    plt.ylabel('Fraction of Positives (Actual PD)', fontsize=12)
    plt.title('PD Calibration Curve: Quantile Strategy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    for i, (x, y) in enumerate(zip(prob_pred, prob_true)):
        plt.annotate(f'Bin {i+1}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
    plt.show()

def generate_score_pd_mapping(df, score_col='SCORE', target_col='TARGET', n_bins=10):
    """
    Maps scores to risk grades and calculates real-world PD for each bucket.
    """
    df = df.copy()
    # High score = Low risk, so G1 is the highest score group
    df['Risk_Grade'] = pd.qcut(df[score_col], q=n_bins, labels=[f'G{i}' for i in range(1, n_bins+1)][::-1])
    
    mapping = df.groupby('Risk_Grade').agg(
        Min_Score=(score_col, 'min'),
        Max_Score=(score_col, 'max'),
        Actual_PD=(target_col, 'mean'),
        Customer_Count=(target_col, 'count')
    ).reset_index()
    
    mapping['Actual_PD_%'] = (mapping['Actual_PD'] * 100).round(2)
    return mapping

def generate_score_pd_mapping(df, score_col='SCORE', target_col='TARGET', n_bins=10):
    """
    Creates a mapping table between score ranges and average Probability of Default (PD).
    """
    df = df.copy()
    
    df = df.sort_values(by=score_col, ascending=False)
    
    # Create Risk Grades (G1: Best to G10: Worst)
    df['Risk_Grade'] = pd.qcut(df[score_col], q=n_bins, labels=[f'G{i}' for i in range(1, n_bins+1)])
    
    mapping_table = df.groupby('Risk_Grade').agg(
        Min_Score=(score_col, 'min'),
        Max_Score=(score_col, 'max'),
        Avg_PD=(target_col, 'mean'),
        Customer_Count=(target_col, 'count')
    ).reset_index()
    
    mapping_table['Avg_PD_%'] = (mapping_table['Avg_PD'] * 100).round(2)
    
    return mapping_table

def calculate_brier_score(y_true, y_prob):
    """
    Calculates the Brier Score to measure the accuracy of probabilistic predictions.
    Lower is better (0 is perfect).
    """
    return brier_score_loss(y_true, y_prob)

def calculate_expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE). 
    Measures the average gap between confidence and accuracy across bins.
    Lower is better (0 is perfect).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        bin_weight = np.sum(in_bin) / total_samples
        
        if bin_weight > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            confidence_in_bin = np.mean(y_prob[in_bin])
            ece += bin_weight * np.abs(accuracy_in_bin - confidence_in_bin)
            
    return ece