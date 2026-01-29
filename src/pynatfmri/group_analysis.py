"""
Group-level analysis module for pynatfmri.

This module provides functions for analyzing inter-subject functional connectivity
at the group level, including group comparisons and brain-behavior correlations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
from scipy import stats
import warnings


def load_participants_data(bids_root: Union[str, Path]) -> pd.DataFrame:
    """
    Load participants.tsv file from BIDS directory.
    
    Parameters
    ----------
    bids_root : str or Path
        Path to BIDS root directory
    
    Returns
    -------
    participants_df : DataFrame
        Participant demographics and behavioral data
    """
    bids_root = Path(bids_root)
    participants_file = bids_root / "participants.tsv"
    
    if not participants_file.exists():
        raise FileNotFoundError(f"participants.tsv not found: {participants_file}")
    
    df = pd.read_csv(participants_file, sep="\t")
    
    # Ensure participant_id column has proper format
    if 'participant_id' in df.columns:
        # Add 'sub-' prefix if not present
        df['participant_id'] = df['participant_id'].astype(str).apply(
            lambda x: x if x.startswith('sub-') else f'sub-{x}'
        )
    
    return df


def load_isfc_results(
    bids_root: Union[str, Path],
    session: str = "01",
    task: str = "rest",
    run: str = "01",
    space: str = "MNI152NLin6Asym",
    seg: str = "schaefer200",
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Load ISFC results from BIDS derivatives directory.
    
    Parameters
    ----------
    bids_root : str or Path
        Path to BIDS root directory
    session : str, default: "01"
        Session identifier
    task : str, default: "rest"
        Task identifier
    run : str, default: "01"
        Run identifier
    space : str, default: "MNI152NLin6Asym"
        Space label
    seg : str, default: "schaefer200"
        Segmentation label
    
    Returns
    -------
    isfc_data : dict
        Dictionary mapping subject IDs to their ISFC DataFrames
    roi_pairs : list of str
        List of ROI pair names (columns in DataFrames)
    """
    bids_root = Path(bids_root)
    deriv_dir = bids_root / "derivatives" / "pynatfmri_isc"
    
    if not deriv_dir.exists():
        raise FileNotFoundError(f"ISFC derivatives not found: {deriv_dir}")
    
    isfc_data = {}
    roi_pairs = None
    
    # Find all subject directories
    subject_dirs = sorted([d for d in deriv_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("sub-")])
    
    for sub_dir in subject_dirs:
        subject_id = sub_dir.name
        
        # Construct filepath
        func_dir = sub_dir / f"ses-{session}" / "func"
        filename = (f"{subject_id}_ses-{session}_task-{task}_run-{run}_"
                   f"space-{space}_seg-{seg}_stat-isc_relmat.tsv")
        filepath = func_dir / filename
        
        if not filepath.exists():
            warnings.warn(f"ISFC file not found for {subject_id}: {filepath}")
            continue
        
        # Load ISFC data
        df = pd.read_csv(filepath, sep="\t")
        isfc_data[subject_id] = df
        
        # Extract ROI pair names from first subject
        if roi_pairs is None:
            roi_pairs = [col for col in df.columns if col != 'partner_subject']
    
    if len(isfc_data) == 0:
        raise ValueError("No ISFC data files found")
    
    return isfc_data, roi_pairs


def compute_mean_isfc_per_subject(
    isfc_data: Dict[str, pd.DataFrame],
    roi_pairs: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute mean ISFC for each subject across all their partners.
    
    Parameters
    ----------
    isfc_data : dict
        Dictionary mapping subject IDs to their ISFC DataFrames
    roi_pairs : list of str, optional
        Specific ROI pairs to include. If None, uses all.
    
    Returns
    -------
    mean_isfc_df : DataFrame
        Subject ID, mean ISFC per ROI pair, and overall mean
    """
    results = []
    
    for subject_id, df in isfc_data.items():
        if roi_pairs is None:
            roi_cols = [col for col in df.columns if col != 'partner_subject']
        else:
            roi_cols = roi_pairs
        
        # Mean across partners for each ROI pair
        mean_values = df[roi_cols].mean(axis=0)
        
        # Overall mean across all ROI pairs and partners
        overall_mean = df[roi_cols].values.flatten()
        overall_mean = np.nanmean(overall_mean)
        
        result = {'participant_id': subject_id, 'mean_isfc_overall': overall_mean}
        result.update({f'mean_{roi}': mean_values[roi] for roi in roi_cols})
        
        results.append(result)
    
    return pd.DataFrame(results)


def compare_groups(
    isfc_data: Dict[str, pd.DataFrame],
    participants_df: pd.DataFrame,
    group_column: str = 'group',
    roi_pairs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare ISFC between groups using t-tests.
    
    Parameters
    ----------
    isfc_data : dict
        Dictionary mapping subject IDs to their ISFC DataFrames
    participants_df : DataFrame
        Participant demographics with group assignments
    group_column : str, default: 'group'
        Column name for group labels
    roi_pairs : list of str, optional
        Specific ROI pairs to compare
    
    Returns
    -------
    comparison_df : DataFrame
        Statistical comparison results per ROI pair
    """
    # Get mean ISFC per subject
    mean_isfc = compute_mean_isfc_per_subject(isfc_data, roi_pairs)
    
    # Merge with participant data
    merged = mean_isfc.merge(participants_df, on='participant_id')
    
    if group_column not in merged.columns:
        raise ValueError(f"Group column '{group_column}' not found in participants data")
    
    groups = merged[group_column].unique()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, found {len(groups)}: {groups}")
    
    group1, group2 = sorted(groups)
    
    # Get ROI pair columns
    if roi_pairs is None:
        roi_cols = [col for col in mean_isfc.columns 
                   if col.startswith('mean_') and col != 'mean_isfc_overall']
    else:
        roi_cols = [f'mean_{roi}' for roi in roi_pairs]
    
    # Add overall mean
    roi_cols = ['mean_isfc_overall'] + roi_cols
    
    results = []
    
    for roi_col in roi_cols:
        group1_vals = merged[merged[group_column] == group1][roi_col].dropna()
        group2_vals = merged[merged[group_column] == group2][roi_col].dropna()
        
        # T-test
        t_stat, p_val = stats.ttest_ind(group1_vals, group2_vals)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(group1_vals) - 1) * group1_vals.std()**2 + 
             (len(group2_vals) - 1) * group2_vals.std()**2) / 
            (len(group1_vals) + len(group2_vals) - 2)
        )
        cohens_d = (group1_vals.mean() - group2_vals.mean()) / pooled_std
        
        results.append({
            'roi_pair': roi_col.replace('mean_', ''),
            'group1': group1,
            'group1_mean': group1_vals.mean(),
            'group1_std': group1_vals.std(),
            'group1_n': len(group1_vals),
            'group2': group2,
            'group2_mean': group2_vals.mean(),
            'group2_std': group2_vals.std(),
            'group2_n': len(group2_vals),
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
        })
    
    comparison_df = pd.DataFrame(results)
    
    # FDR correction
    from scipy.stats import false_discovery_control
    comparison_df['p_value_fdr'] = false_discovery_control(comparison_df['p_value'])
    
    return comparison_df


def brain_behavior_correlation(
    isfc_data: Dict[str, pd.DataFrame],
    participants_df: pd.DataFrame,
    behavior_column: str,
    group_column: Optional[str] = None,
    roi_pairs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Correlate ISFC with behavioral measures.
    
    Parameters
    ----------
    isfc_data : dict
        Dictionary mapping subject IDs to their ISFC DataFrames
    participants_df : DataFrame
        Participant demographics and behavioral data
    behavior_column : str
        Column name for behavioral measure
    group_column : str, optional
        If provided, compute correlations separately per group
    roi_pairs : list of str, optional
        Specific ROI pairs to analyze
    
    Returns
    -------
    correlation_df : DataFrame
        Correlation results per ROI pair (and group if specified)
    """
    # Get mean ISFC per subject
    mean_isfc = compute_mean_isfc_per_subject(isfc_data, roi_pairs)
    
    # Merge with participant data
    merged = mean_isfc.merge(participants_df, on='participant_id')
    
    if behavior_column not in merged.columns:
        raise ValueError(f"Behavior column '{behavior_column}' not found")
    
    # Get ROI pair columns
    if roi_pairs is None:
        roi_cols = [col for col in mean_isfc.columns 
                   if col.startswith('mean_') and col != 'mean_isfc_overall']
    else:
        roi_cols = [f'mean_{roi}' for roi in roi_pairs]
    
    # Add overall mean
    roi_cols = ['mean_isfc_overall'] + roi_cols
    
    results = []
    
    # Determine groups to analyze
    if group_column is not None and group_column in merged.columns:
        groups = [(g, merged[merged[group_column] == g]) for g in merged[group_column].unique()]
        groups.append(('all', merged))  # Add overall correlation
    else:
        groups = [('all', merged)]
    
    for group_name, group_df in groups:
        for roi_col in roi_cols:
            # Remove NaN values
            valid_data = group_df[[roi_col, behavior_column]].dropna()
            
            if len(valid_data) < 3:
                continue
            
            # Pearson correlation
            r, p = stats.pearsonr(valid_data[roi_col], valid_data[behavior_column])
            
            results.append({
                'roi_pair': roi_col.replace('mean_', ''),
                'group': group_name,
                'n': len(valid_data),
                'r': r,
                'p_value': p,
                'behavior': behavior_column,
            })
    
    correlation_df = pd.DataFrame(results)
    
    # FDR correction within each group
    for group_name in correlation_df['group'].unique():
        mask = correlation_df['group'] == group_name
        p_vals = correlation_df.loc[mask, 'p_value'].values
        from scipy.stats import false_discovery_control
        correlation_df.loc[mask, 'p_value_fdr'] = false_discovery_control(p_vals)
    
    return correlation_df


def within_vs_between_group_isfc(
    isfc_data: Dict[str, pd.DataFrame],
    participants_df: pd.DataFrame,
    group_column: str = 'group',
    roi_pairs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare within-group vs between-group ISFC.
    
    Parameters
    ----------
    isfc_data : dict
        Dictionary mapping subject IDs to their ISFC DataFrames
    participants_df : DataFrame
        Participant demographics with group assignments
    group_column : str, default: 'group'
        Column name for group labels
    roi_pairs : list of str, optional
        Specific ROI pairs to analyze
    
    Returns
    -------
    results_df : DataFrame
        Within-group and between-group ISFC statistics
    """
    # Create subject-to-group mapping
    subject_groups = participants_df.set_index('participant_id')[group_column].to_dict()
    
    groups = participants_df[group_column].unique()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, found {len(groups)}")
    
    group1, group2 = sorted(groups)
    
    # Determine ROI pairs to analyze
    if roi_pairs is None:
        first_subject = list(isfc_data.keys())[0]
        roi_pairs = [col for col in isfc_data[first_subject].columns 
                    if col != 'partner_subject']
    
    results = []
    
    for roi_pair in roi_pairs:
        within_group1 = []
        within_group2 = []
        between_groups = []
        
        for subject_id, df in isfc_data.items():
            if subject_id not in subject_groups:
                continue
            
            subject_group = subject_groups[subject_id]
            
            for _, row in df.iterrows():
                partner_id = row['partner_subject']
                if partner_id not in subject_groups:
                    continue
                
                partner_group = subject_groups[partner_id]
                isfc_val = row[roi_pair]
                
                if np.isnan(isfc_val):
                    continue
                
                # Categorize pair type
                if subject_group == partner_group:
                    if subject_group == group1:
                        within_group1.append(isfc_val)
                    else:
                        within_group2.append(isfc_val)
                else:
                    between_groups.append(isfc_val)
        
        # Compute statistics
        result = {
            'roi_pair': roi_pair,
            f'within_{group1}_mean': np.mean(within_group1) if within_group1 else np.nan,
            f'within_{group1}_std': np.std(within_group1) if within_group1 else np.nan,
            f'within_{group1}_n': len(within_group1),
            f'within_{group2}_mean': np.mean(within_group2) if within_group2 else np.nan,
            f'within_{group2}_std': np.std(within_group2) if within_group2 else np.nan,
            f'within_{group2}_n': len(within_group2),
            'between_groups_mean': np.mean(between_groups) if between_groups else np.nan,
            'between_groups_std': np.std(between_groups) if between_groups else np.nan,
            'between_groups_n': len(between_groups),
        }
        
        # Statistical tests
        if within_group1 and between_groups:
            t1, p1 = stats.ttest_ind(within_group1, between_groups)
            result[f'{group1}_vs_between_t'] = t1
            result[f'{group1}_vs_between_p'] = p1
        
        if within_group2 and between_groups:
            t2, p2 = stats.ttest_ind(within_group2, between_groups)
            result[f'{group2}_vs_between_t'] = t2
            result[f'{group2}_vs_between_p'] = p2
        
        results.append(result)
    
    return pd.DataFrame(results)


def save_group_analysis_results(
    results: pd.DataFrame,
    output_path: Union[str, Path],
    description: str = "Group analysis results"
):
    """
    Save group analysis results to TSV file.
    
    Parameters
    ----------
    results : DataFrame
        Analysis results
    output_path : str or Path
        Output file path
    description : str
        Description for output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(output_path, sep="\t", index=False)
    print(f"✓ Saved {description}: {output_path}")
