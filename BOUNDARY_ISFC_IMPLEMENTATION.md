# Boundary-Centered ISFC Implementation Summary

## Overview

I've created a new `boundary_isfc.py` module in the pynatfmri library that implements boundary-centered inter-subject functional connectivity (ISFC) analysis between posterior hippocampus and medial PFC during event boundaries.

## Module Location
- **File**: `src/pynatfmri/boundary_isfc.py`
- **Updated**: `src/pynatfmri/__init__.py` to expose the module

## Key Functions

### 1. **`compute_boundary_isfc()`** - Main analysis function
Orchestrates the full analysis pipeline:
- Loads event boundaries from `events.csv`
- Loads participant metadata from `participants.tsv`
- Extracts ROI timeseries from BIDS derivatives for all subjects
- Computes ISFC for each event-subject pair
- Returns DataFrame with all results

**Parameters:**
- `events_file`: Path to events.csv (subject_id, group, event_id, accuracy, boundary_seconds)
- `participants_file`: Path to participants.tsv (participant_id, group, ...)
- `bids_deriv_root`: Path to BIDS derivatives/gimmefMRI folder
- `hpc_roi`: Hippocampus ROI name (default: "LR_postJHipp_200")
- `mpfc_roi`: mPFC ROI name (default: "LR_mPFC_200")
- `window_duration`: Window size in seconds (default: 24.0)
- `tr`: Repetition time in seconds (default: 2.0)
- `offset_seconds`: Offset from boundary/midpoint to center window (default: 6.0)

### 2. **`compute_boundary_isfc_single_event()`** - Per-event computation
Computes ISFC values for a single event-subject pair:
- Extracts 24-second windows centered at +6s post-boundary (or post-midpoint for null)
- Handles NaN values (scrubbed volumes) gracefully
- Computes LOSO (leave-one-subject-out) within-group averages
- Computes cross-group averages
- Performs correlations with Fisher z-transformation

**Connectivity types computed:**
- `hpc_x_mpfc_within_boundary/null`: HPC × LOSO mPFC (same age group)
- `mpfc_x_hpc_within_boundary/null`: mPFC × LOSO HPC (same age group)
- `hpc_x_mpfc_cross_boundary/null`: HPC × other age group mPFC
- `mpfc_x_hpc_cross_boundary/null`: mPFC × other age group HPC

### 3. **`compute_null_window_center()`** - Null window calculation
Computes midpoint between consecutive event boundaries for null control windows.

### 4. **Helper Functions:**
- `load_boundary_events()`: Parse events.csv
- `load_participants()`: Parse participants.tsv
- `extract_roi_timeseries()`: Load ROI data from BIDS derivative TSV files
- `extract_window()`: Extract temporal window from timeseries
- `compute_correlation_skip_nans()`: Compute Pearson correlation, skip NaNs
- `compute_loso_group_mean()`: Compute leave-one-subject-out group average
- `save_boundary_isfc()`: Save results to CSV

## Output DataFrame Structure

Each row represents one event-subject pair with columns:

**Metadata:**
- `subject_id`: Subject identifier (e.g., "sub-101")
- `group`: Age group (YA or OA)
- `event_id`: Event identifier
- `accuracy`: Event memory accuracy (0 or 1)
- `boundary_seconds`: Event boundary time in seconds

**Boundary Window Connectivity (centered at +6s post-boundary):**
- `hpc_x_mpfc_within_boundary`: HPC→mPFC within-group ISFC (Fisher z)
- `mpfc_x_hpc_within_boundary`: mPFC→HPC within-group ISFC (Fisher z)
- `hpc_x_mpfc_cross_boundary`: HPC→mPFC cross-group ISFC (Fisher z)
- `mpfc_x_hpc_cross_boundary`: mPFC→HPC cross-group ISFC (Fisher z)
- `n_clean_boundary`: Number of clean timepoints in boundary window

**Null Window Connectivity (centered at +6s post-midpoint):**
- `hpc_x_mpfc_within_null`: HPC→mPFC within-group ISFC (Fisher z)
- `mpfc_x_hpc_within_null`: mPFC→HPC within-group ISFC (Fisher z)
- `hpc_x_mpfc_cross_null`: HPC→mPFC cross-group ISFC (Fisher z)
- `mpfc_x_hpc_cross_null`: mPFC→HPC cross-group ISFC (Fisher z)
- `n_clean_null`: Number of clean timepoints in null window

## Design Decisions

### 1. **Window Parameters**
- **Size**: 24 seconds (12 TRs @ 2s TR) - provides sufficient data for reliable correlation
- **Boundary window center**: +6s post-boundary - accounts for HRF delay
- **Null window center**: +6s post-event midpoint (between consecutive boundaries)
- **Offset**: 6s is added to both boundary and midpoint times

### 2. **NaN Handling**
- Scrubbed volumes (marked as empty cells in source data) are **skipped** during correlation computation
- Group averages use `np.nanmean()` to exclude NaNs at each timepoint
- The number of clean points is tracked and included in output

### 3. **ISFC Computation**
- **Within-group ISFC**: Correlate subject ROI with LOSO mean of same-age group ROI
- **Cross-group ISFC**: Correlate subject ROI with mean of other-age group ROI
- **Both directional pairs**: HPC→mPFC and mPFC→HPC computed separately
- All correlations Fisher z-transformed before output

### 4. **Null Window Strategy**
- True midpoints between consecutive boundaries used as control
- Same window duration (24s) and offset (+6s) as boundary windows
- Allows direct comparison of boundary vs. control connectivity

## Example Usage

```python
from pynatfmri import boundary_isfc

results_df = boundary_isfc.compute_boundary_isfc(
    events_file="path/to/events.csv",
    participants_file="path/to/participants.tsv",
    bids_deriv_root="path/to/BIDS/derivatives/gimmefMRI",
)

# Save results
boundary_isfc.save_boundary_isfc(results_df, "boundary_isfc_results.csv")

# Prepare for mixed-effects models
print(results_df.head())
```

## Next Steps for Analysis

1. **Mixed-effects models**: Use ISFC values to predict recall by age group
2. **Exploratory analyses**: 
   - Sliding window approach for peri-boundary dynamics
   - Lag-ISFC to test lead/lag relationships
   - FIR/deconvolution models for HRF sensitivity
3. **Covariate analysis**: Add motion, tSNR, perceptual covariates if needed

## Notes

- Boundaries are identical across subjects (same stimulus/movie for all)
- Event boundaries are sorted chronologically; null windows use inter-boundary midpoints
- Cross-group connectivity uses simple mean (not LOSO) since subjects are from different age groups
- Fisher z-transformation ensures normal distribution for statistical modeling
