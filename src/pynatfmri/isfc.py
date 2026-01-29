"""
Inter-subject functional connectivity module for pynatfmri.

This module provides functions for computing inter-subject functional connectivity (ISFC)
from BIDS-formatted timeseries derivatives.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Tuple
import warnings


def load_timeseries_from_bids(
    bids_root: Union[str, Path],
    derivative_name: str = "xcp_d",
    subjects: Optional[List[str]] = None,
    session: str = "01",
    task: str = "rest",
    run: str = "01",
    space: str = "MNI152NLin6Asym",
    seg: str = "schaefer200",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load timeseries data from BIDS derivatives directory.
    
    Parameters
    ----------
    bids_root : str or Path
        Path to BIDS root directory
    derivative_name : str, default: "xcp_d"
        Name of derivatives folder (e.g., "xcp_d", "gimmefMRI")
    subjects : list of str, optional
        List of subject IDs to load. If None, loads all available subjects.
    session : str, default: "01"
        Session identifier
    task : str, default: "rest"
        Task identifier
    run : str, default: "01"
        Run identifier
    space : str, default: "MNI152NLin6Asym"
        Space label
    seg : str, default: "schaefer200"
        Segmentation/atlas label
    
    Returns
    -------
    data : ndarray (n_TRs x n_ROIs x n_subjects)
        Timeseries data array
    subject_ids : list of str
        Subject IDs in order
    roi_names : list of str
        ROI/column names
    """
    bids_root = Path(bids_root)
    deriv_dir = bids_root / "derivatives" / derivative_name
    
    if not deriv_dir.exists():
        raise FileNotFoundError(f"Derivatives directory not found: {deriv_dir}")
    
    # Find all subjects if not specified
    if subjects is None:
        subject_dirs = sorted([d for d in deriv_dir.iterdir() 
                              if d.is_dir() and d.name.startswith("sub-")])
        subjects = [d.name for d in subject_dirs]
    else:
        # Ensure subjects have "sub-" prefix
        subjects = [s if s.startswith("sub-") else f"sub-{s}" for s in subjects]
    
    if len(subjects) == 0:
        raise ValueError(f"No subjects found in {deriv_dir}")
    
    # Load data for each subject
    data_list = []
    valid_subjects = []
    roi_names = None
    
    for sub in subjects:
        # Construct filepath
        func_dir = deriv_dir / sub / f"ses-{session}" / "func"
        filename = (f"{sub}_ses-{session}_task-{task}_run-{run}_"
                   f"space-{space}_seg-{seg}_stat-mean_timeseries.tsv")
        filepath = func_dir / filename
        
        if not filepath.exists():
            warnings.warn(f"Timeseries file not found for {sub}: {filepath}")
            continue
        
        # Load TSV
        df = pd.read_csv(filepath, sep="\t")
        
        # Store ROI names from first subject
        if roi_names is None:
            roi_names = df.columns.tolist()
        
        # Check consistency
        if df.columns.tolist() != roi_names:
            warnings.warn(f"ROI names differ for {sub}, skipping")
            continue
        
        data_list.append(df.values)
        valid_subjects.append(sub)
    
    if len(data_list) == 0:
        raise ValueError(f"No valid timeseries data found for any subjects")
    
    # Check that all subjects have same number of timepoints
    n_trs = [d.shape[0] for d in data_list]
    if len(set(n_trs)) > 1:
        warnings.warn(f"Subjects have different numbers of timepoints: {set(n_trs)}")
        # Truncate to minimum length
        min_trs = min(n_trs)
        data_list = [d[:min_trs, :] for d in data_list]
    
    # Stack into 3D array: n_TRs x n_ROIs x n_subjects
    data = np.dstack(data_list)
    
    return data, valid_subjects, roi_names


def pairwise_isfc(
    data: np.ndarray,
    roi_indices: Optional[Union[List[int], np.ndarray]] = None
) -> np.ndarray:
    """
    Compute pairwise inter-subject functional connectivity.
    
    For selected ROIs, compute pairwise correlation between all pairs of subjects.
    Returns an n_pairs x n_ROI_pairs matrix.
    
    Parameters
    ----------
    data : ndarray (n_TRs x n_ROIs x n_subjects)
        Timeseries data
    roi_indices : list or array, optional
        Indices of ROIs to compute ISFC for. If None, uses all ROIs.
    
    Returns
    -------
    isfc_matrix : ndarray (n_pairs x n_ROI_pairs)
        Pairwise ISFC values. n_pairs = n_subjects*(n_subjects-1)/2
        n_ROI_pairs = n_selected_ROIs * (n_selected_ROIs - 1) / 2
    """
    n_TRs, n_ROIs, n_subjects = data.shape
    
    # Select ROIs
    if roi_indices is None:
        roi_indices = np.arange(n_ROIs)
    else:
        roi_indices = np.array(roi_indices)
    
    selected_data = data[:, roi_indices, :]
    n_selected = len(roi_indices)
    
    # Compute pairwise ISFCs
    from itertools import combinations
    
    # All subject pairs
    subject_pairs = list(combinations(range(n_subjects), 2))
    n_pairs = len(subject_pairs)
    
    # All ROI pairs
    roi_pairs = list(combinations(range(n_selected), 2))
    n_roi_pairs = len(roi_pairs)
    
    # Initialize output matrix
    isfc_matrix = np.zeros((n_pairs, n_roi_pairs))
    
    # Compute correlation for each subject pair and ROI pair
    for pair_idx, (sub1, sub2) in enumerate(subject_pairs):
        for roi_pair_idx, (roi1, roi2) in enumerate(roi_pairs):
            # Correlation between roi1 in sub1 and roi2 in sub2
            ts1 = selected_data[:, roi1, sub1]
            ts2 = selected_data[:, roi2, sub2]
            
            # Find valid (non-NaN) timepoints in both series
            valid_mask = ~(np.isnan(ts1) | np.isnan(ts2))
            
            if np.sum(valid_mask) < 3:
                # Not enough valid timepoints
                isfc_matrix[pair_idx, roi_pair_idx] = np.nan
            else:
                # Pearson correlation on valid timepoints only
                corr = np.corrcoef(ts1[valid_mask], ts2[valid_mask])[0, 1]
                
                # Fisher-Z transform
                if np.abs(corr) >= 1.0:
                    # Handle edge cases
                    fisher_z = np.nan
                else:
                    fisher_z = np.arctanh(corr)
                
                isfc_matrix[pair_idx, roi_pair_idx] = fisher_z
    
    return isfc_matrix


def symmetric_pairwise_isfc(
    data: np.ndarray,
    roi_indices: Optional[Union[List[int], np.ndarray]] = None
) -> np.ndarray:
    """
    Compute symmetric pairwise inter-subject functional connectivity.
    
    For selected ROIs, compute pairwise correlation between all pairs of subjects.
    Averages correlation in both directions (sub1->sub2 and sub2->sub1).
    
    Parameters
    ----------
    data : ndarray (n_TRs x n_ROIs x n_subjects)
        Timeseries data
    roi_indices : list or array, optional
        Indices of ROIs to compute ISFC for. If None, uses all ROIs.
    
    Returns
    -------
    isfc_matrix : ndarray (n_pairs x n_ROI_pairs)
        Symmetric pairwise ISFC values
    """
    n_TRs, n_ROIs, n_subjects = data.shape
    
    # Select ROIs
    if roi_indices is None:
        roi_indices = np.arange(n_ROIs)
    else:
        roi_indices = np.array(roi_indices)
    
    selected_data = data[:, roi_indices, :]
    n_selected = len(roi_indices)
    
    from itertools import combinations
    
    # All subject pairs
    subject_pairs = list(combinations(range(n_subjects), 2))
    n_pairs = len(subject_pairs)
    
    # All ROI pairs
    roi_pairs = list(combinations(range(n_selected), 2))
    n_roi_pairs = len(roi_pairs)
    
    # Initialize output matrix
    isfc_matrix = np.zeros((n_pairs, n_roi_pairs))
    
    # Compute symmetric correlation for each subject pair and ROI pair
    for pair_idx, (sub1, sub2) in enumerate(subject_pairs):
        for roi_pair_idx, (roi1, roi2) in enumerate(roi_pairs):
            # Correlation in both directions
            ts1_roi1 = selected_data[:, roi1, sub1]
            ts2_roi2 = selected_data[:, roi2, sub2]
            
            # Find valid timepoints for forward correlation
            valid_forward = ~(np.isnan(ts1_roi1) | np.isnan(ts2_roi2))
            
            ts1_roi2 = selected_data[:, roi2, sub1]
            ts2_roi1 = selected_data[:, roi1, sub2]
            
            # Find valid timepoints for backward correlation
            valid_backward = ~(np.isnan(ts1_roi2) | np.isnan(ts2_roi1))
            
            # Compute correlations if enough valid data
            if np.sum(valid_forward) < 3 and np.sum(valid_backward) < 3:
                isfc_matrix[pair_idx, roi_pair_idx] = np.nan
            else:
                fisher_values = []
                
                if np.sum(valid_forward) >= 3:
                    corr_forward = np.corrcoef(ts1_roi1[valid_forward], 
                                              ts2_roi2[valid_forward])[0, 1]
                    if np.abs(corr_forward) < 1.0:
                        fisher_values.append(np.arctanh(corr_forward))
                
                if np.sum(valid_backward) >= 3:
                    corr_backward = np.corrcoef(ts1_roi2[valid_backward], 
                                               ts2_roi1[valid_backward])[0, 1]
                    if np.abs(corr_backward) < 1.0:
                        fisher_values.append(np.arctanh(corr_backward))
                
                # Average Fisher-Z values (or use single value if only one direction valid)
                if len(fisher_values) > 0:
                    isfc_matrix[pair_idx, roi_pair_idx] = np.mean(fisher_values)
                else:
                    isfc_matrix[pair_idx, roi_pair_idx] = np.nan
    
    return isfc_matrix


def save_isfc_to_bids(
    isfc_matrix: np.ndarray,
    subject_pairs: List[Tuple[str, str]],
    roi_pair_names: List[Tuple[str, str]],
    bids_root: Union[str, Path],
    session: str = "01",
    task: str = "rest",
    run: str = "01",
    space: str = "MNI152NLin6Asym",
    seg: str = "schaefer200",
):
    """
    Save ISFC results to BIDS derivatives directory.
    
    Creates one TSV file per subject with their ISFC values to all other subjects.
    
    Parameters
    ----------
    isfc_matrix : ndarray (n_pairs x n_ROI_pairs)
        ISFC values
    subject_pairs : list of tuples
        List of (sub1, sub2) pairs corresponding to rows of isfc_matrix
    roi_pair_names : list of tuples
        List of (roi1, roi2) names corresponding to columns of isfc_matrix
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
        Segmentation/atlas label
    """
    bids_root = Path(bids_root)
    deriv_dir = bids_root / "derivatives" / "pynatfmri_isc"
    
    # Create column names from ROI pairs
    col_names = [f"{roi1}_{roi2}" for roi1, roi2 in roi_pair_names]
    
    # Get unique subjects
    all_subjects = set()
    for sub1, sub2 in subject_pairs:
        all_subjects.add(sub1)
        all_subjects.add(sub2)
    
    # Save one file per subject with their ISFC values
    for subject in sorted(all_subjects):
        # Find all pairs involving this subject
        pair_indices = [i for i, (s1, s2) in enumerate(subject_pairs)
                       if s1 == subject or s2 == subject]
        
        if len(pair_indices) == 0:
            continue
        
        # Extract ISFC values for this subject
        subject_isfc = isfc_matrix[pair_indices, :]
        
        # Get partner subjects
        partners = [s2 if s1 == subject else s1 
                   for i, (s1, s2) in enumerate(subject_pairs) 
                   if i in pair_indices]
        
        # Create DataFrame with explicit column creation
        # Start with partner_subject column
        df_data = {'partner_subject': partners}
        
        # Add each ROI pair column
        for col_idx, col_name in enumerate(col_names):
            df_data[col_name] = subject_isfc[:, col_idx]
        
        df = pd.DataFrame(df_data)
        
        # Create output directory
        func_dir = deriv_dir / subject / f"ses-{session}" / "func"
        func_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct filename
        filename = (f"{subject}_ses-{session}_task-{task}_run-{run}_"
                   f"space-{space}_seg-{seg}_stat-isc_relmat.tsv")
        filepath = func_dir / filename
        
        # Save TSV
        df.to_csv(filepath, sep="\t", index=False)
        print(f"Saved: {filepath}")
