# Bug Fix: Shape Mismatch in Boundary ISFC Correlation

## Problem

Error when running `compute_boundary_isfc.py`:
```
ValueError: operands could not be broadcast together with shapes (12,) (1012,)
```

This occurred in `compute_correlation_skip_nans()` when trying to correlate:
- **Windowed data** (12 timepoints - extracted boundary/null window)
- **Full timeseries group mean** (1012 timepoints - entire timeseries)

## Root Cause

The LOSO group mean was being computed from full-length timeseries, but then we tried to correlate it with windowed (extracted) timeseries from individual subjects.

The flow was:
1. Extract 12-timepoint window for subject's HPC
2. Compute LOSO mean from **full 1012-timepoint timeseries** of group
3. Try to correlate (12,) with (1012,) → Shape mismatch error

## Solution

**Extract windows for all group members FIRST, then compute LOSO mean from windowed data.**

### Changes Made

**File**: `src/pynatfmri/boundary_isfc.py`

#### 1. Within-Group ISFC (LOSO)
**Before**: Computed LOSO mean from full timeseries
```python
# Wrong: computing from full timeseries
loso_same_hpc_boundary = compute_loso_group_mean(same_group_hpc_list, idx)
# where same_group_hpc_list = [1012-point timeseries, 1012-point timeseries, ...]
```

**After**: Extract windows for all group members, then compute LOSO mean
```python
# Right: extract windows first
same_group_hpc_boundary_windows = []  # Store extracted windows
for sub_id in same_group_subjects:
    hpc_full = hpc_data_dict[sub_id]
    hpc_win, _ = extract_window(hpc_full, boundary_center, ...)  # Extract 24s window
    same_group_hpc_boundary_windows.append(hpc_win)

# Now all timeseries are same length (12 timepoints)
loso_same_hpc_boundary = compute_loso_group_mean(same_group_hpc_boundary_windows, idx)
```

#### 2. Cross-Group ISFC
**Before**: Computed mean from full timeseries
```python
# Wrong: computing from full timeseries
other_hpc_mean_boundary = np.nanmean(np.column_stack(other_group_hpc_list), axis=1)
```

**After**: Extract windows for other group subjects first
```python
# Right: extract windows first
other_group_hpc_boundary = []
for sub_id in other_group_subjects:
    hpc_full = hpc_data_dict[sub_id]
    hpc_win, _ = extract_window(hpc_full, boundary_center, ...)
    other_group_hpc_boundary.append(hpc_win)

# Now compute mean of windowed data
other_hpc_mean_boundary = np.nanmean(np.column_stack(other_group_hpc_boundary), axis=1)
```

#### 3. Null Window Handling
Added proper initialization and conditional logic:
```python
# Initialize null variables
loso_same_hpc_null = None
loso_same_mpfc_null = None

# Only compute if null window is valid
if null_window_center is not None and len(same_group_hpc_null_windows) >= 2:
    loso_same_hpc_null = compute_loso_group_mean(same_group_hpc_null_windows, idx)
    loso_same_mpfc_null = compute_loso_group_mean(same_group_mpfc_null_windows, idx)

# Use conditional correlation computation
if loso_same_mpfc_null is not None:
    hpc_x_mpfc_within_null, _ = compute_correlation_skip_nans(hpc_null, loso_same_mpfc_null)
else:
    hpc_x_mpfc_within_null = None
```

## Data Flow After Fix

```
1. Subject's timeseries (full): [1012 points]
   ↓ Extract window
2. Subject's windowed data: [12 points]

3. All same-group subjects' windows: [[12 points], [12 points], [12 points], ...]
   ↓ Compute LOSO mean (exclude subject)
4. LOSO group mean: [12 points]

5. Subject window [12 points] ⊗ LOSO mean [12 points]
   ↓ Correlate → success!
6. Fisher z-transformed correlation
```

## Verification

The fix ensures:
- ✅ All timeseries being correlated have matching lengths
- ✅ LOSO computation works on windowed data only
- ✅ Cross-group means computed from windowed data
- ✅ Null window logic properly handles missing data
- ✅ No undefined variable references

## Testing

To verify the fix works:
```bash
conda activate pynatfmri
cd natfMRI/scripts
python compute_boundary_isfc.py
```

Should now complete without shape mismatch errors.

## Implementation Details

**File modified**: `src/pynatfmri/boundary_isfc.py`
**Function**: `compute_boundary_isfc_single_event()`
**Lines changed**: ~150 lines refactored for proper window extraction ordering

**Key principle**: Always extract windows at the same time for all subjects being compared, ensuring temporal alignment and matching data lengths.
