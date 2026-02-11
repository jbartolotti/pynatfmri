# Boundary ISFC: Quick Start Guide

## Installation & Setup

1. **Install/update pynatfmri** with the new boundary_isfc module:
   ```bash
   cd path/to/pynatfmri
   pip install -e .  # editable install
   ```

2. **Verify the module loads**:
   ```python
   from pynatfmri import boundary_isfc
   print(boundary_isfc.__doc__)
   ```

## Running the Analysis

### Minimal Example
```python
from pathlib import Path
from pynatfmri import boundary_isfc

# Define paths (adjust as needed)
natfmri_root = Path(r"p:\IRB_STUDY00149390_A015\MR_Data\Connectivity\natfMRI")
events_file = natfmri_root / "events.csv"
participants_file = natfmri_root / "A015_BIDS" / "participants.tsv"
bids_deriv_root = natfmri_root / "A015_BIDS" / "derivatives" / "gimmefMRI"

# Run analysis with defaults
results_df = boundary_isfc.compute_boundary_isfc(
    events_file=events_file,
    participants_file=participants_file,
    bids_deriv_root=bids_deriv_root,
)

# Save results
boundary_isfc.save_boundary_isfc(results_df, natfmri_root / "boundary_isfc_results.csv")

# Inspect results
print(results_df.shape)
print(results_df.columns.tolist())
print(results_df.head())
```

### With Custom Parameters
```python
results_df = boundary_isfc.compute_boundary_isfc(
    events_file=events_file,
    participants_file=participants_file,
    bids_deriv_root=bids_deriv_root,
    hpc_roi="LR_postJHipp_200",    # default
    mpfc_roi="LR_mPFC_200",         # default
    window_duration=24.0,           # seconds (12 TRs @ 2s TR)
    tr=2.0,                         # seconds
    offset_seconds=6.0,             # from boundary/midpoint to window center
)
```

## Output Interpretation

### DataFrame Columns

**Metadata:**
- `subject_id`: e.g., "sub-101"
- `group`: "YA" or "OA"
- `event_id`: Event number (1-35 in your data)
- `accuracy`: 0 or 1 (memory accuracy for this event)
- `boundary_seconds`: Event onset time in seconds

**Primary Analysis (Boundary Window):**
```
Boundary window: [boundary_time + 0s, boundary_time + 24s] 
Center: boundary_time + 6s
Duration: 12 timepoints (24s @ 2s TR)

Within-group ISFC (LOSO):
- hpc_x_mpfc_within_boundary:  HPC × mPFC(same-group LOSO)  [Fisher z]
- mpfc_x_hpc_within_boundary:  mPFC × HPC(same-group LOSO)  [Fisher z]

Cross-group ISFC:
- hpc_x_mpfc_cross_boundary:   HPC × mPFC(other-group mean)  [Fisher z]
- mpfc_x_hpc_cross_boundary:   mPFC × HPC(other-group mean)  [Fisher z]

Data quality:
- n_clean_boundary:            Number of non-NaN timepoints in window
```

**Control Window (Null):**
```
Null window: [midpoint_between_boundaries + 0s, + 24s]
Center: midpoint + 6s
Purpose: Control for non-boundary-specific connectivity

Same connectivity measures as above:
- hpc_x_mpfc_within_null, mpfc_x_hpc_within_null
- hpc_x_mpfc_cross_null, mpfc_x_hpc_cross_null
- n_clean_null

Expected: Null ISFC should be similar across groups and
          smaller than boundary ISFC if effect is boundary-specific
```

### Data Quality Checks

```python
# Check for missing values
print(results_df.isnull().sum())

# Examine data cleaning
print(f"Mean clean points (boundary): {results_df['n_clean_boundary'].mean():.1f}")
print(f"Mean clean points (null):     {results_df['n_clean_null'].mean():.1f}")

# Check expected ranges for Fisher z
print(f"HPC×mPFC boundary range: [{results_df['hpc_x_mpfc_within_boundary'].min():.3f}, "
      f"{results_df['hpc_x_mpfc_within_boundary'].max():.3f}]")

# Sample sizes by group
print("\nSample sizes:")
print(results_df.groupby('group')['subject_id'].nunique())
print(results_df.groupby('group')['event_id'].nunique())
```

## Next: Mixed-Effects Modeling

Once you have the boundary ISFC results, typical next steps:

### 1. Prepare Data for Modeling
```python
# Fisher z values are already computed
# Just need to scale/center if desired

# Check distributions
import matplotlib.pyplot as plt
results_df['hpc_x_mpfc_within_boundary'].hist(bins=30)
plt.xlabel("Fisher z")
plt.ylabel("Frequency")
plt.show()
```

### 2. Simple Mixed-Effects Model Example (using statsmodels)
```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fit mixed-effects model: accuracy ~ ISFC + age_group + (1|subject)
model = smf.mixedlm(
    "accuracy ~ C(group) * hpc_x_mpfc_within_boundary",
    data=results_df,
    groups=results_df["subject_id"]
)
result = model.fit()
print(result.summary())
```

### 3. Compare Boundary vs Null
```python
# Test whether boundary ISFC > null ISFC
results_df['boundary_vs_null'] = (
    results_df['hpc_x_mpfc_within_boundary'] - 
    results_df['hpc_x_mpfc_within_null']
)

# Positive values indicate stronger boundary-locked coupling
print(f"Mean difference (boundary - null): {results_df['boundary_vs_null'].mean():.3f}")
print(f"t-test p-value: {scipy.stats.ttest_1samp(results_df['boundary_vs_null'], 0)[1]:.4f}")
```

## Troubleshooting

### "Subject XXX not found in participants list"
- Check: Subject ID format in events.csv (e.g., 101) vs participants.tsv (e.g., sub-101)
- Solution: Both should use same format (or prepend "sub-" automatically, which code does)

### "ROI 'LR_postJHipp_200' not found in timeseries"
- Check: Timeseries file column names with:
  ```python
  import pandas as pd
  ts = pd.read_csv("path/to/timeseries.tsv", sep="\t", nrows=1)
  print(ts.columns.tolist())
  ```
- Solution: Update `hpc_roi` parameter with correct column name

### "No valid correlations computed"
- Check: n_clean values - may indicate excessive scrubbing
- Solution: Check source timeseries for quality issues

### "Insufficient same-group subjects"
- Likely issue: Small sample size in one age group or incomplete data
- Solution: Check participants.tsv has correct group assignments

## Files Created

1. **Module**: `src/pynatfmri/boundary_isfc.py` - Main implementation
2. **Examples**: `example_boundary_isfc.py` - Template script
3. **Documentation**:
   - `BOUNDARY_ISFC_IMPLEMENTATION.md` - Overview & function reference
   - `BOUNDARY_ISFC_TECHNICAL_DESIGN.md` - Design rationale & architecture
   - `BOUNDARY_ISFC_QUICKSTART.md` - This file

## Next Steps for Analysis

1. ✅ Run `compute_boundary_isfc()` to generate results
2. ⬜ Verify output with data quality checks
3. ⬜ Exploratory analysis: Plot ISFC distributions by group
4. ⬜ Mixed-effects models: ISFC predicting recall × age interaction
5. ⬜ Sensitivity analyses:
   - Different window sizes/offsets
   - Within- vs cross-group comparisons
   - Boundary vs null contrast
6. ⬜ (Optional) Advanced approaches:
   - Lag-ISFC to test temporal dynamics
   - FIR deconvolution for HRF robustness
   - Perceptual covariate inclusion

---

**Questions?** Review the technical design document or reach out with specific issues.
