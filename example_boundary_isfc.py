#!/usr/bin/env python
"""
Example script for computing boundary-centered ISFC.
"""

from pathlib import Path
from pynatfmri import boundary_isfc

# Define paths
natfmri_root = Path(r"p:\IRB_STUDY00149390_A015\MR_Data\Connectivity\natfMRI")
events_file = natfmri_root / "events.csv"
participants_file = natfmri_root / "A015_BIDS" / "participants.tsv"
bids_deriv_root = natfmri_root / "A015_BIDS" / "derivatives" / "gimmefMRI"

# Compute boundary ISFC
results_df = boundary_isfc.compute_boundary_isfc(
    events_file=events_file,
    participants_file=participants_file,
    bids_deriv_root=bids_deriv_root,
    hpc_roi="LR_postJHipp_200",
    mpfc_roi="LR_mPFC_200",
    window_duration=24.0,  # 24 seconds
    tr=2.0,  # 2 second TR
    offset_seconds=6.0,  # Center window at +6s post-boundary
)

print(f"\nResults shape: {results_df.shape}")
print(f"\nFirst few rows:")
print(results_df.head())

print(f"\nColumn names:")
print(results_df.columns.tolist())

print(f"\nData types:")
print(results_df.dtypes)

# Save results
output_file = natfmri_root / "boundary_isfc_results.csv"
boundary_isfc.save_boundary_isfc(results_df, output_file)
