"""
Boundary-centered inter-subject functional connectivity (ISFC) module for pynatfmri.

This module computes inter-subject functional connectivity between ROI pairs
(e.g., posterior hippocampus and mPFC) during event boundaries and null control windows.
Supports leave-one-out (LOSO) averaging within and across age groups.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
import warnings


def load_boundary_events(
    events_file: Union[str, Path],
) -> pd.DataFrame:
    """
    Load event boundaries from CSV file.
    
    Parameters
    ----------
    events_file : str or Path
        Path to events.csv file with columns: subject_id, group, event_id, accuracy, boundary_seconds
    
    Returns
    -------
    events_df : pd.DataFrame
        Events dataframe
    """
    events_df = pd.read_csv(events_file)
    return events_df


def load_participants(
    participants_file: Union[str, Path],
) -> pd.DataFrame:
    """
    Load participant metadata.
    
    Parameters
    ----------
    participants_file : str or Path
        Path to participants.tsv file with columns: participant_id, group, ...
    
    Returns
    -------
    participants_df : pd.DataFrame
        Participants dataframe
    """
    participants_df = pd.read_csv(participants_file, sep="\t")
    return participants_df


def extract_roi_timeseries(
    bids_deriv_root: Union[str, Path],
    subject_id: str,
    roi_names: List[str],
    session: str = "01",
) -> Dict[str, np.ndarray]:
    """
    Extract specific ROI timeseries for a subject from BIDS derivatives.
    
    Parameters
    ----------
    bids_deriv_root : str or Path
        Path to BIDS derivatives/gimmefMRI folder
    subject_id : str
        Subject ID (e.g., "sub-101")
    roi_names : list of str
        ROI names to extract (must match column names in timeseries TSV)
    session : str, default: "01"
        Session identifier
    
    Returns
    -------
    roi_data : dict
        Dictionary mapping roi_name -> timeseries (1D array)
    """
    bids_deriv_root = Path(bids_deriv_root)
    func_dir = bids_deriv_root / subject_id / f"ses-{session}" / "func"
    
    # Find the timeseries file
    ts_files = list(func_dir.glob("*_stat-mean_timeseries.tsv"))
    if len(ts_files) == 0:
        raise FileNotFoundError(f"No timeseries file found in {func_dir}")
    if len(ts_files) > 1:
        warnings.warn(f"Multiple timeseries files found in {func_dir}, using first")
    
    ts_file = ts_files[0]
    
    # Load data
    df = pd.read_csv(ts_file, sep="\t")
    
    # Extract requested ROIs
    roi_data = {}
    for roi_name in roi_names:
        if roi_name not in df.columns:
            raise ValueError(f"ROI '{roi_name}' not found in timeseries. Available: {df.columns.tolist()}")
        roi_data[roi_name] = df[roi_name].values
    
    return roi_data


def extract_window(
    timeseries: np.ndarray,
    center_time: float,
    window_duration: float = 24.0,
    tr: float = 2.0,
) -> Tuple[np.ndarray, int]:
    """
    Extract a centered window from a timeseries.
    
    Parameters
    ----------
    timeseries : ndarray (n_timepoints,)
        1D timeseries
    center_time : float
        Center time in seconds
    window_duration : float, default: 24.0
        Window duration in seconds
    tr : float, default: 2.0
        Repetition time in seconds
    
    Returns
    -------
    window : ndarray
        Windowed timeseries (values may include NaNs from scrubbed volumes)
    n_clean : int
        Number of non-NaN values in the window
    """
    # Convert times to indices
    center_idx = int(np.round(center_time / tr))
    half_duration = window_duration / 2.0
    start_idx = int(np.round((center_time - half_duration) / tr))
    end_idx = int(np.round((center_time + half_duration) / tr))
    
    # Clip to valid range
    start_idx = max(0, start_idx)
    end_idx = min(len(timeseries), end_idx)
    
    if start_idx >= end_idx:
        raise ValueError(f"Invalid window indices: start={start_idx}, end={end_idx}")
    
    window = timeseries[start_idx:end_idx]
    n_clean = np.sum(~np.isnan(window))
    
    return window, n_clean


def compute_correlation_skip_nans(
    ts1: np.ndarray,
    ts2: np.ndarray,
    min_valid_points: int = 5,
) -> Tuple[Optional[float], int]:
    """
    Compute Pearson correlation between two timeseries, skipping NaN values.
    
    Parameters
    ----------
    ts1, ts2 : ndarray
        Timeseries (may contain NaNs)
    min_valid_points : int, default: 5
        Minimum number of valid datapoints required to compute correlation
    
    Returns
    -------
    fisher_z : float or None
        Fisher z-transformed correlation, or None if insufficient valid data
    n_valid : int
        Number of valid datapoints used
    """
    # Find valid (non-NaN) timepoints in both series
    valid_mask = ~(np.isnan(ts1) | np.isnan(ts2))
    n_valid = np.sum(valid_mask)
    
    if n_valid < min_valid_points:
        return None, n_valid
    
    # Compute Pearson correlation
    ts1_clean = ts1[valid_mask]
    ts2_clean = ts2[valid_mask]
    
    # Use np.corrcoef for correlation
    if len(ts1_clean) < 2:
        return None, n_valid
    
    corr = np.corrcoef(ts1_clean, ts2_clean)[0, 1]
    
    # Handle edge cases
    if np.isnan(corr) or np.abs(corr) >= 1.0:
        return None, n_valid
    
    # Fisher z-transform
    fisher_z = np.arctanh(corr)
    
    return fisher_z, n_valid


def compute_loso_group_mean(
    roi_data_list: List[np.ndarray],
    exclude_idx: int,
) -> np.ndarray:
    """
    Compute leave-one-subject-out (LOSO) group mean for a ROI.
    
    Handles NaN values by computing mean across non-NaN values at each timepoint.
    
    Parameters
    ----------
    roi_data_list : list of ndarray
        List of timeseries from each subject in the group (should be windowed/same length)
    exclude_idx : int
        Index of subject to exclude
    
    Returns
    -------
    group_mean : ndarray
        LOSO group mean timeseries
    """
    # Select data excluding the target subject
    other_subjects = [roi_data_list[i] for i in range(len(roi_data_list)) if i != exclude_idx]
    
    # Stack into 2D array: n_timepoints x n_others
    stacked = np.column_stack(other_subjects)
    
    # Compute mean across subjects, ignoring NaNs at each timepoint
    group_mean = np.nanmean(stacked, axis=1)
    
    return group_mean


def compute_null_window_center(
    event_idx: int,
    boundary_times: List[float],
) -> Optional[float]:
    """
    Compute the center time for a null window at the midpoint between consecutive boundaries.
    
    Parameters
    ----------
    event_idx : int
        Index of the current event in the boundary_times list
    boundary_times : list of float
        All boundary times in seconds (sorted)
    
    Returns
    -------
    null_center : float or None
        Center time for the null window, or None if no next boundary exists
    """
    if event_idx < len(boundary_times) - 1:
        # Midpoint between current and next boundary
        current_boundary = boundary_times[event_idx]
        next_boundary = boundary_times[event_idx + 1]
        null_center = (current_boundary + next_boundary) / 2.0
        return null_center
    else:
        # No next boundary - skip null window or use a fixed offset
        return None


def compute_boundary_isfc_single_event(
    subject_idx: int,
    subject_id: str,
    group: str,
    event_id: int,
    boundary_seconds: float,
    accuracy: int,
    all_subjects: List[str],
    all_groups: List[str],
    hpc_data_dict: Dict[str, np.ndarray],
    mpfc_data_dict: Dict[str, np.ndarray],
    roi_names: Tuple[str, str],
    window_duration: float = 24.0,
    tr: float = 2.0,
    offset_seconds: float = 6.0,
    null_window_center: Optional[float] = None,
) -> Dict:
    """
    Compute boundary-centered ISFC for a single event in a single subject.
    
    Computes both boundary windows (+offset to +offset+duration) and null windows
    (at event midpoints between consecutive boundaries).
    
    Parameters
    ----------
    subject_idx : int
        Index of this subject in the group list
    subject_id : str
        Subject identifier
    group : str
        Age group (e.g., "YA", "OA")
    event_id : int
        Event identifier
    boundary_seconds : float
        Event boundary time in seconds
    accuracy : int
        Event accuracy (0 or 1)
    all_subjects : list of str
        All subject IDs in entire cohort
    all_groups : list of str
        Age group for each subject in all_subjects
    hpc_data_dict : dict
        Maps subject_id -> hippocampus timeseries
    mpfc_data_dict : dict
        Maps subject_id -> mPFC timeseries
    roi_names : tuple of str
        (hpc_roi_name, mpfc_roi_name)
    window_duration : float, default: 24.0
        Window duration in seconds
    tr : float, default: 2.0
        Repetition time in seconds
    offset_seconds : float, default: 6.0
        Offset from boundary/midpoint to center the window at
    null_window_center : float or None, default: None
        Center time for null window. If None, null window computation is skipped.
    
    Returns
    -------
    result : dict
        Dictionary with connectivity values and metadata
    """
    hpc_roi_name, mpfc_roi_name = roi_names
    
    # Get this subject's timeseries
    try:
        hpc_ts = hpc_data_dict[subject_id]
        mpfc_ts = mpfc_data_dict[subject_id]
    except KeyError:
        return None
    
    # Extract boundary window
    boundary_center = boundary_seconds + offset_seconds
    hpc_boundary, hpc_n_clean_boundary = extract_window(
        hpc_ts, boundary_center, window_duration, tr
    )
    mpfc_boundary, mpfc_n_clean_boundary = extract_window(
        mpfc_ts, boundary_center, window_duration, tr
    )
    
    # Extract null window (if center is provided)
    if null_window_center is not None:
        null_center = null_window_center + offset_seconds
        hpc_null, hpc_n_clean_null = extract_window(
            hpc_ts, null_center, window_duration, tr
        )
        mpfc_null, mpfc_n_clean_null = extract_window(
            mpfc_ts, null_center, window_duration, tr
        )
    else:
        # No valid null window - use NaN placeholders
        hpc_null = np.full_like(hpc_boundary, np.nan)
        mpfc_null = np.full_like(mpfc_boundary, np.nan)
        hpc_n_clean_null = 0
        mpfc_n_clean_null = 0
    
    # Find other subjects in same age group (for within-group LOSO)
    same_group_indices = [i for i, g in enumerate(all_groups) if g == group]
    same_group_subjects = [all_subjects[i] for i in same_group_indices]
    
    # Find other subjects in different age group (for cross-group)
    other_group_indices = [i for i, g in enumerate(all_groups) if g != group]
    other_group_subjects = [all_subjects[i] for i in other_group_indices]
    
    # Get timeseries for same-group subjects
    same_group_hpc_full = []
    same_group_mpfc_full = []
    for sub_id in same_group_subjects:
        hpc = hpc_data_dict.get(sub_id)
        mpfc = mpfc_data_dict.get(sub_id)
        if hpc is not None and mpfc is not None:
            same_group_hpc_full.append(hpc)
            same_group_mpfc_full.append(mpfc)
    
    if len(same_group_hpc_full) < 2 or len(same_group_mpfc_full) < 2:
        warnings.warn(f"Insufficient same-group subjects for {subject_id}")
        return None
    
    # Find subject index within same-group list
    same_group_subject_idx = None
    for idx, sub_id in enumerate(same_group_subjects):
        if sub_id == subject_id:
            same_group_subject_idx = idx
            break
    
    if same_group_subject_idx is None:
        warnings.warn(f"Subject {subject_id} not found in same-group list")
        return None
    
    # Extract windows for all same-group subjects
    same_group_hpc_boundary_windows = []
    same_group_mpfc_boundary_windows = []
    same_group_hpc_null_windows = []
    same_group_mpfc_null_windows = []
    
    valid_subject_indices = []  # Track which subjects have valid data
    
    for idx, sub_id in enumerate(same_group_subjects):
        hpc_full = hpc_data_dict.get(sub_id)
        mpfc_full = mpfc_data_dict.get(sub_id)
        
        if hpc_full is None or mpfc_full is None:
            continue
        
        # Extract boundary window
        hpc_win, _ = extract_window(hpc_full, boundary_center, window_duration, tr)
        mpfc_win, _ = extract_window(mpfc_full, boundary_center, window_duration, tr)
        same_group_hpc_boundary_windows.append(hpc_win)
        same_group_mpfc_boundary_windows.append(mpfc_win)
        
        # Extract null window (if applicable)
        if null_window_center is not None:
            null_center = null_window_center + offset_seconds
            hpc_null_win, _ = extract_window(hpc_full, null_center, window_duration, tr)
            mpfc_null_win, _ = extract_window(mpfc_full, null_center, window_duration, tr)
            same_group_hpc_null_windows.append(hpc_null_win)
            same_group_mpfc_null_windows.append(mpfc_null_win)
        
        valid_subject_indices.append(idx)
    
    if len(same_group_hpc_boundary_windows) < 2:
        warnings.warn(f"Insufficient subjects with valid data for {subject_id}")
        return None
    
    # Compute LOSO group means from WINDOWED data
    # Find the index of target subject within valid subjects
    target_idx_in_valid = None
    for vidx, sidx in enumerate(valid_subject_indices):
        if sidx == same_group_subject_idx:
            target_idx_in_valid = vidx
            break
    
    if target_idx_in_valid is None:
        warnings.warn(f"Target subject {subject_id} not in valid subject list")
        return None
    
    loso_same_hpc_boundary = compute_loso_group_mean(same_group_hpc_boundary_windows, target_idx_in_valid)
    loso_same_mpfc_boundary = compute_loso_group_mean(same_group_mpfc_boundary_windows, target_idx_in_valid)
    
    # Compute LOSO group means for null window (if applicable)
    loso_same_hpc_null = None
    loso_same_mpfc_null = None
    
    if null_window_center is not None and len(same_group_hpc_null_windows) >= 2:
        loso_same_hpc_null = compute_loso_group_mean(same_group_hpc_null_windows, target_idx_in_valid)
        loso_same_mpfc_null = compute_loso_group_mean(same_group_mpfc_null_windows, target_idx_in_valid)
    
    # Compute within-group connectivity for boundary window
    hpc_x_mpfc_within_boundary, n_valid_hb1 = compute_correlation_skip_nans(
        hpc_boundary, loso_same_mpfc_boundary
    )
    mpfc_x_hpc_within_boundary, n_valid_hb2 = compute_correlation_skip_nans(
        mpfc_boundary, loso_same_hpc_boundary
    )
    
    # Compute within-group connectivity for null window
    if loso_same_mpfc_null is not None:
        hpc_x_mpfc_within_null, n_valid_nb1 = compute_correlation_skip_nans(
            hpc_null, loso_same_mpfc_null
        )
        mpfc_x_hpc_within_null, n_valid_nb2 = compute_correlation_skip_nans(
            mpfc_null, loso_same_hpc_null
        )
    else:
        hpc_x_mpfc_within_null = None
        mpfc_x_hpc_within_null = None
    
    # Compute cross-group connectivity (if other group exists)
    if len(other_group_subjects) >= 2:
        # Extract windows for other group subjects
        other_group_hpc_boundary = []
        other_group_mpfc_boundary = []
        other_group_hpc_null = []
        other_group_mpfc_null = []
        
        for sub_id in other_group_subjects:
            hpc_full = hpc_data_dict.get(sub_id)
            mpfc_full = mpfc_data_dict.get(sub_id)
            
            if hpc_full is None or mpfc_full is None:
                continue
            
            # Extract boundary window
            hpc_win, _ = extract_window(hpc_full, boundary_center, window_duration, tr)
            mpfc_win, _ = extract_window(mpfc_full, boundary_center, window_duration, tr)
            other_group_hpc_boundary.append(hpc_win)
            other_group_mpfc_boundary.append(mpfc_win)
            
            # Extract null window (if applicable)
            if null_window_center is not None:
                null_center = null_window_center + offset_seconds
                hpc_null_win, _ = extract_window(hpc_full, null_center, window_duration, tr)
                mpfc_null_win, _ = extract_window(mpfc_full, null_center, window_duration, tr)
                other_group_hpc_null.append(hpc_null_win)
                other_group_mpfc_null.append(mpfc_null_win)
        
        if len(other_group_hpc_boundary) >= 2 and len(other_group_mpfc_boundary) >= 2:
            # Compute mean across other group (simple mean, not LOSO since different group)
            other_hpc_mean_boundary = np.nanmean(np.column_stack(other_group_hpc_boundary), axis=1)
            other_mpfc_mean_boundary = np.nanmean(np.column_stack(other_group_mpfc_boundary), axis=1)
            
            hpc_x_mpfc_cross_boundary, n_valid_cb1 = compute_correlation_skip_nans(
                hpc_boundary, other_mpfc_mean_boundary
            )
            mpfc_x_hpc_cross_boundary, n_valid_cb2 = compute_correlation_skip_nans(
                mpfc_boundary, other_hpc_mean_boundary
            )
            
            if null_window_center is not None and len(other_group_hpc_null) >= 2:
                other_hpc_mean_null = np.nanmean(np.column_stack(other_group_hpc_null), axis=1)
                other_mpfc_mean_null = np.nanmean(np.column_stack(other_group_mpfc_null), axis=1)
                
                hpc_x_mpfc_cross_null, n_valid_cn1 = compute_correlation_skip_nans(
                    hpc_null, other_mpfc_mean_null
                )
                mpfc_x_hpc_cross_null, n_valid_cn2 = compute_correlation_skip_nans(
                    mpfc_null, other_hpc_mean_null
                )
            else:
                hpc_x_mpfc_cross_null = None
                mpfc_x_hpc_cross_null = None
        else:
            hpc_x_mpfc_cross_boundary = None
            mpfc_x_hpc_cross_boundary = None
            hpc_x_mpfc_cross_null = None
            mpfc_x_hpc_cross_null = None
    else:
        hpc_x_mpfc_cross_boundary = None
        mpfc_x_hpc_cross_boundary = None
        hpc_x_mpfc_cross_null = None
        mpfc_x_hpc_cross_null = None
    
    result = {
        'subject_id': subject_id,
        'group': group,
        'event_id': event_id,
        'accuracy': accuracy,
        'boundary_seconds': boundary_seconds,
        
        # Boundary window connectivity
        'hpc_x_mpfc_within_boundary': hpc_x_mpfc_within_boundary,
        'mpfc_x_hpc_within_boundary': mpfc_x_hpc_within_boundary,
        'hpc_x_mpfc_cross_boundary': hpc_x_mpfc_cross_boundary,
        'mpfc_x_hpc_cross_boundary': mpfc_x_hpc_cross_boundary,
        'n_clean_boundary': min(hpc_n_clean_boundary, mpfc_n_clean_boundary),
        
        # Null window connectivity
        'hpc_x_mpfc_within_null': hpc_x_mpfc_within_null,
        'mpfc_x_hpc_within_null': mpfc_x_hpc_within_null,
        'hpc_x_mpfc_cross_null': hpc_x_mpfc_cross_null,
        'mpfc_x_hpc_cross_null': mpfc_x_hpc_cross_null,
        'n_clean_null': min(hpc_n_clean_null, mpfc_n_clean_null),
    }
    
    return result


def compute_boundary_isfc(
    events_file: Union[str, Path],
    participants_file: Union[str, Path],
    bids_deriv_root: Union[str, Path],
    hpc_roi: str = "LR_postJHipp_200",
    mpfc_roi: str = "LR_mPFC_200",
    window_duration: float = 24.0,
    tr: float = 2.0,
    offset_seconds: float = 6.0,
) -> pd.DataFrame:
    """
    Compute boundary-centered ISFC for all events and subjects.
    
    Parameters
    ----------
    events_file : str or Path
        Path to events.csv file
    participants_file : str or Path
        Path to participants.tsv file
    bids_deriv_root : str or Path
        Path to BIDS derivatives/gimmefMRI folder
    hpc_roi : str, default: "LR_postJHipp_200"
        Hippocampus ROI column name
    mpfc_roi : str, default: "LR_mPFC_200"
        mPFC ROI column name
    window_duration : float, default: 24.0
        Window duration in seconds
    tr : float, default: 2.0
        Repetition time in seconds
    offset_seconds : float, default: 6.0
        Offset from boundary/midpoint to center window at
    
    Returns
    -------
    results_df : pd.DataFrame
        Dataframe with connectivity values for each event-subject pair
    """
    # Load metadata
    events_df = load_boundary_events(events_file)
    participants_df = load_participants(participants_file)
    
    # Create mapping from subject_id to group
    subject_to_group = dict(zip(participants_df['participant_id'], participants_df['group']))
    
    # Get all unique subjects
    all_subjects_list = sorted(participants_df['participant_id'].unique())
    all_groups_list = [subject_to_group[sub] for sub in all_subjects_list]
    
    # Get unique boundary times in order (for computing null window midpoints)
    unique_boundaries = sorted(events_df['boundary_seconds'].unique())
    
    # Create mapping from boundary_seconds to index for null window computation
    boundary_to_idx = {b: i for i, b in enumerate(unique_boundaries)}
    
    # Load all subject timeseries
    print("Loading timeseries data...")
    hpc_data_dict = {}
    mpfc_data_dict = {}
    
    for subject_id in all_subjects_list:
        try:
            roi_data = extract_roi_timeseries(
                bids_deriv_root, subject_id, [hpc_roi, mpfc_roi]
            )
            hpc_data_dict[subject_id] = roi_data[hpc_roi]
            mpfc_data_dict[subject_id] = roi_data[mpfc_roi]
            print(f"  Loaded {subject_id}")
        except Exception as e:
            warnings.warn(f"Failed to load {subject_id}: {e}")
    
    # Compute ISFC for each event
    print(f"Computing boundary ISFC for {len(events_df)} event-subject pairs...")
    results = []
    
    for idx, row in events_df.iterrows():
        subject_id = f"sub-{row['subject_id']}"
        group = row['group']
        event_id = row['event_id']
        boundary_seconds = row['boundary_seconds']
        accuracy = row['accuracy']
        
        # Find subject index
        if subject_id not in all_subjects_list:
            warnings.warn(f"Subject {subject_id} not found in participants list")
            continue
        
        subject_idx = all_subjects_list.index(subject_id)
        
        # Compute null window center
        boundary_idx = boundary_to_idx[boundary_seconds]
        null_center = compute_null_window_center(boundary_idx, unique_boundaries)
        
        # Compute ISFC
        result = compute_boundary_isfc_single_event(
            subject_idx=subject_idx,
            subject_id=subject_id,
            group=group,
            event_id=event_id,
            boundary_seconds=boundary_seconds,
            accuracy=accuracy,
            all_subjects=all_subjects_list,
            all_groups=all_groups_list,
            hpc_data_dict=hpc_data_dict,
            mpfc_data_dict=mpfc_data_dict,
            roi_names=(hpc_roi, mpfc_roi),
            window_duration=window_duration,
            tr=tr,
            offset_seconds=offset_seconds,
            null_window_center=null_center,
        )
        
        if result is not None:
            results.append(result)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(events_df)} events")
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    return results_df


def save_boundary_isfc(
    results_df: pd.DataFrame,
    output_file: Union[str, Path],
):
    """
    Save boundary ISFC results to file.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from compute_boundary_isfc
    output_file : str or Path
        Path to output CSV file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
