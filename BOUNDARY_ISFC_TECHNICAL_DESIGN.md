# Boundary ISFC: Technical Design & Rationale

## Problem Statement
Compute inter-subject functional connectivity (ISFC) between posterior hippocampus and medial PFC during event boundaries, with goal of predicting memory accuracy in mixed-effects models. Need to account for age differences (YA vs OA) and control for false positives using null windows.

## Solution Architecture

### Data Flow
```
events.csv + participants.tsv
     ↓
[Load metadata & group assignments]
     ↓
Extract ROI timeseries for all subjects
[HPC_data_dict, mPFC_data_dict]
     ↓
For each event-subject pair:
  - Extract boundary window (boundary_time + 6s offset)
  - Extract null window (inter-boundary midpoint + 6s offset)
  - Compute LOSO group means
  - Correlate with NaN handling
  - Fisher z-transform
     ↓
Output DataFrame (ready for mixed-effects models)
```

## Key Design Choices & Justification

### 1. Window Geometry

**Choice**: 24s window centered at +6s post-boundary (or post-midpoint)
- Equivalent to 12 TRs (with TR=2s)
- Offset of +6s accounts for HRF delay

**Rationale**:
- Previous communication established that +2 to +8s was insufficient (only 3-4 TRs)
- 12 timepoints provides adequate degrees of freedom for correlation estimation
- +6s post-boundary aligns with typical HRF peak timing for stimulus-locked fMRI
- Same window for null condition ensures matched statistical power

**Alternative considered**: 
- Event-locked FIR or HRF convolution → deferred to later sensitivity analysis
- Rationale: Fixed offset sufficient for primary hypothesis; HRF uncertainty can be tested post-hoc

### 2. Null Window Strategy

**Choice**: Midpoint between consecutive boundaries with +6s offset
- Uses actual inter-boundary intervals from the data
- Mid-event location controls for stimulus-driven (non-boundary) connectivity
- Same duration & offset ensures fair comparison

**Implementation detail**:
```python
boundary_idx = boundary_to_idx[boundary_seconds]
null_center = (boundaries[idx] + boundaries[idx+1]) / 2.0
null_window_center = null_center + offset_seconds  # +6s offset
```

**Why this matters**: Prevents null windows from inadvertently capturing a nearby boundary.

### 3. NaN Handling Strategy

**Choice**: Skip NaNs during correlation, track number of clean points

**Procedure**:
- Extract full window (including NaNs from scrubbing)
- When computing correlation: create valid_mask excluding NaNs
- Compute Pearson r only on valid timepoints
- If < 5 valid points available, return NaN for that correlation

**Code**:
```python
valid_mask = ~(np.isnan(ts1) | np.isnan(ts2))
n_valid = np.sum(valid_mask)
if n_valid >= 5:
    r = np.corrcoef(ts1[valid_mask], ts2[valid_mask])[0, 1]
    fisher_z = np.arctanh(r)
```

**Alternatives considered**:
- Interpolate NaNs → biases correlations, inflates degrees of freedom
- Exclude entire window if any NaN → loses data unnecessarily
- **Selected approach**: Honest accounting of available data, tracked in output

**Why track n_clean in output?**
- Allows post-hoc investigation of quality/power differences
- Enables sample-size-weighted analyses if needed
- Diagnostic tool for identifying problematic subjects/sessions

### 4. ISFC Computation: Within-Group

**Choice**: Leave-one-subject-out (LOSO) averaging

**Procedure**:
```python
# For subject i in age group G:
group_G_minus_i = [subject_j in G where j ≠ i]
group_mean = nanmean(timeseries_of_group_G_minus_i)  # at each timepoint
isfc_i = corr(subject_i_ROI, group_mean)
```

**Why LOSO?**
- Standard in ISFC literature (reduces circularity / overfitting)
- Eliminates dependency that would arise if subject i's data influenced the mean it's correlated against
- Fair comparison across groups (all subjects treated identically)

**Implementation detail**: 
```python
def compute_loso_group_mean(roi_data_list, exclude_idx):
    other_subjects = [roi_data_list[i] for i in range(len(roi_data_list)) if i != exclude_idx]
    stacked = np.column_stack(other_subjects)  # shape: n_timepoints x n_others
    group_mean = np.nanmean(stacked, axis=1)   # accounts for NaNs at each TR
    return group_mean
```

### 5. ISFC Computation: Cross-Group

**Choice**: Simple mean (not LOSO) of other age group

**Rationale**:
- Other-group subjects are not part of subject i's group definition
- No circularity issue (subject i cannot be in other age group)
- LOSO adds complexity without benefit here

**Procedure**:
```python
other_group_mean = np.nanmean(np.column_stack(other_group_timeseries), axis=1)
isfc_cross = corr(subject_i_ROI, other_group_mean)
```

### 6. Directional Correlations

**Choice**: Compute BOTH HPC→mPFC and mPFC→HPC

**Rationale**:
- Even though correlations are symmetric, modeling framework may benefit from both
- Later mixed-effects models can choose which directionality to focus on
- Provides flexibility for exploratory analyses (e.g., testing asymmetric coupling)

**Output columns**:
- `hpc_x_mpfc_within_boundary`: HPC timeseries × mPFC group mean
- `mpfc_x_hpc_within_boundary`: mPFC timeseries × HPC group mean
- (Same for cross-group and null windows)

### 7. Fisher Z-Transformation

**Choice**: Apply to all output correlations

**Procedure**:
```python
if np.abs(r) < 1.0:
    z = np.arctanh(r)  # or 0.5 * ln((1+r)/(1-r))
```

**Why necessary?**
- Pearson r is not normally distributed (especially at extremes)
- Fisher z approximates normal N(z; μ_z, 1/n) for large n
- Essential for downstream mixed-effects models (linear regression assumes normality)
- Standard practice in ISFC literature

### 8. Metadata Preservation

**Output includes**:
- `subject_id`, `group`, `event_id`, `accuracy`: Links to original data/behavior
- `boundary_seconds`: Temporal reference point
- `n_clean_boundary`, `n_clean_null`: Data quality / statistical power

**Rationale**: 
- All information needed to run mixed-effects models linking ISFC → recall
- Enables post-hoc quality control and diagnostics
- Supports subgroup analyses (e.g., by accuracy, age subranges)

## Efficiency & Scalability

### Computational Strategy
1. **Load all ROI data once** → dictionary lookup during event loop (avoids redundant I/O)
2. **Vectorized operations**: Use numpy for correlation and group averaging
3. **Progress monitoring**: Print every 100 events to track progress
4. **Flexible I/O**: Works with any BIDS-compliant derivatives folder

### Time Complexity
- Loading: O(n_subjects × n_timepoints) - one pass
- ISFC computation: O(n_events × n_subjects × n_ROI_pairs × n_timepoints)
  - For typical dataset (~60 subjects, ~35 events, 24s windows): ~minutes

## Extensibility

The module is designed for future enhancements:

### 1. **Lag-ISFC** (future)
```python
# Add lag parameter: compute cross-correlation at lags [-8s, +8s]
# Would require extending compute_correlation_skip_nans()
```

### 2. **FIR Deconvolution** (future)
```python
# Replace fixed offset window with FIR estimates
# Would fit separate beta per TR around boundaries
```

### 3. **Covariate Integration** (future)
```python
# Add tSNR, motion parameters to output
# Would be loaded alongside ROI data
```

### 4. **ROI Flexibility** (already in place)
```python
# Function accepts any ROI names matching TSV columns
# Easy to swap in anterior HPC, other DMN regions, etc.
```

## Validation Checklist

Before running analyses, verify:
- [ ] events.csv has identical boundaries across subjects
- [ ] participants.tsv lists all subjects with correct group assignments
- [ ] BIDS timeseries files exist for all subjects
- [ ] ROI column names (LR_postJHipp_200, LR_mPFC_200) match TSV headers
- [ ] Timeseries have consistent length (or auto-truncated to minimum)
- [ ] No NaN propagation in group averages (test with sample subject)

## Testing

Example validation commands:
```python
# Quick test on single subject
roi_data = boundary_isfc.extract_roi_timeseries(
    bids_deriv_root, "sub-101", ["LR_postJHipp_200", "LR_mPFC_200"]
)
print(f"HPC shape: {roi_data['LR_postJHipp_200'].shape}")
print(f"NaNs in HPC: {np.sum(np.isnan(roi_data['LR_postJHipp_200']))}")

# Test window extraction
ts = roi_data['LR_postJHipp_200']
window, n_clean = boundary_isfc.extract_window(ts, center_time=50.0, window_duration=24.0, tr=2.0)
print(f"Window size: {len(window)}, Clean points: {n_clean}")

# Test correlation with NaN handling
r, n_valid = boundary_isfc.compute_correlation_skip_nans(ts1, ts2)
print(f"Correlation: {r}, Valid points: {n_valid}")
```
