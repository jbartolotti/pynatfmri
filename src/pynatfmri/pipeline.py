"""
Main pipeline interface for pynatfmri analyses.

This module provides high-level functions for running common fMRI connectivity analyses.
"""

from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
import numpy as np
from itertools import combinations
import pandas as pd

from .isfc import (
    load_timeseries_from_bids,
    symmetric_pairwise_isfc,
    save_isfc_to_bids,
)


class ISFCPipeline:
    """
    Pipeline for inter-subject functional connectivity analysis.
    
    This class provides a high-level interface for loading timeseries data,
    computing ISFC, and saving results in BIDS format.
    
    Parameters
    ----------
    bids_root : str or Path
        Path to BIDS root directory
    derivative_name : str, default: "xcp_d"
        Name of derivatives folder containing timeseries
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
    
    Attributes
    ----------
    data : ndarray
        Loaded timeseries data (n_TRs x n_ROIs x n_subjects)
    subject_ids : list of str
        Subject IDs
    roi_names : list of str
        ROI names
    isfc_matrix : ndarray
        Computed ISFC values
    subject_pairs : list of tuples
        Subject pairs
    roi_pair_names : list of tuples
        ROI pair names
    """
    
    def __init__(
        self,
        bids_root: Union[str, Path],
        derivative_name: str = "xcp_d",
        session: str = "01",
        task: str = "rest",
        run: str = "01",
        space: str = "MNI152NLin6Asym",
        seg: str = "schaefer200",
    ):
        self.bids_root = Path(bids_root)
        self._derivative_name = derivative_name
        self._session = session
        self._task = task
        self._run = run
        self._space = space
        self._seg = seg
        
        # Data attributes
        self.data = None
        self.subject_ids = None
        self.roi_names = None
        self.isfc_matrix = None
        self.subject_pairs = None
        self.roi_pair_names = None
        self.selected_roi_indices = None
    
    def load_data(self, subjects: Optional[List[str]] = None):
        """
        Load timeseries data from BIDS derivatives.
        
        Parameters
        ----------
        subjects : list of str, optional
            List of subject IDs to load. If None, loads all available.
        """
        print(f"Loading timeseries from {self._derivative_name} derivatives...")
        
        self.data, self.subject_ids, self.roi_names = load_timeseries_from_bids(
            bids_root=self.bids_root,
            derivative_name=self._derivative_name,
            subjects=subjects,
            session=self._session,
            task=self._task,
            run=self._run,
            space=self._space,
            seg=self._seg,
        )
        
        print(f"✓ Loaded {len(self.subject_ids)} subjects")
        print(f"  Data shape: {self.data.shape} (n_TRs x n_ROIs x n_subjects)")
        print(f"  ROIs: {len(self.roi_names)}")
        
        return self
    
    def select_rois(
        self,
        roi_indices: Optional[Union[List[int], np.ndarray]] = None,
        roi_names: Optional[List[str]] = None,
    ):
        """
        Select subset of ROIs for analysis.
        
        Parameters
        ----------
        roi_indices : list or array, optional
            Indices of ROIs to use. If None, uses all ROIs.
        roi_names : list of str, optional
            Names of ROIs to use. If provided, overrides roi_indices.
        """
        if self.data is None:
            raise ValueError("Must load data first with load_data()")
        
        if roi_names is not None:
            # Convert ROI names to indices
            roi_indices = [i for i, name in enumerate(self.roi_names) 
                          if name in roi_names]
            if len(roi_indices) == 0:
                raise ValueError(f"No matching ROI names found: {roi_names}")
        
        if roi_indices is None:
            roi_indices = list(range(len(self.roi_names)))
        
        self.selected_roi_indices = roi_indices
        
        selected_names = [self.roi_names[i] for i in roi_indices]
        print(f"✓ Selected {len(roi_indices)} ROIs for analysis")
        print(f"  ROI names: {selected_names[:5]}{'...' if len(selected_names) > 5 else ''}")
        
        return self
    
    def compute_isfc(self, symmetric: bool = True):
        """
        Compute pairwise inter-subject functional connectivity.
        
        Parameters
        ----------
        symmetric : bool, default: True
            If True, compute symmetric ISFC (averaged in both directions).
            If False, compute directional ISFC.
        """
        if self.data is None:
            raise ValueError("Must load data first with load_data()")
        
        if self.selected_roi_indices is None:
            self.select_rois()
        
        print(f"\nComputing {'symmetric ' if symmetric else ''}ISFC...")
        
        if symmetric:
            self.isfc_matrix = symmetric_pairwise_isfc(
                self.data,
                roi_indices=self.selected_roi_indices,
            )
        else:
            from .isfc import pairwise_isfc
            self.isfc_matrix = pairwise_isfc(
                self.data,
                roi_indices=self.selected_roi_indices,
            )
        
        # Generate subject pairs and ROI pair names
        self.subject_pairs = list(combinations(self.subject_ids, 2))
        selected_names = [self.roi_names[i] for i in self.selected_roi_indices]
        self.roi_pair_names = list(combinations(selected_names, 2))
        
        print(f"✓ Computed ISFC matrix: {self.isfc_matrix.shape}")
        print(f"  {len(self.subject_pairs)} subject pairs")
        print(f"  {len(self.roi_pair_names)} ROI pairs")
        
        return self
    
    def print_summary(self, n_samples: int = 3):
        """
        Print summary of ISFC results.
        
        Parameters
        ----------
        n_samples : int, default: 3
            Number of sample results to display
        """
        if self.isfc_matrix is None:
            raise ValueError("Must compute ISFC first with compute_isfc()")
        
        print("\nISFC Summary:")
        print(f"  Mean: {np.nanmean(self.isfc_matrix):.3f}")
        print(f"  Std:  {np.nanstd(self.isfc_matrix):.3f}")
        print(f"  Min:  {np.nanmin(self.isfc_matrix):.3f}")
        print(f"  Max:  {np.nanmax(self.isfc_matrix):.3f}")
        
        print(f"\nSample ISFC values:")
        for i in range(min(n_samples, len(self.subject_pairs))):
            sub1, sub2 = self.subject_pairs[i]
            print(f"  {sub1} <-> {sub2}:")
            for j in range(min(n_samples, len(self.roi_pair_names))):
                roi1, roi2 = self.roi_pair_names[j]
                print(f"    {roi1} <-> {roi2}: {self.isfc_matrix[i, j]:.3f}")
    
    def save_results(self):
        """
        Save ISFC results to BIDS derivatives directory.
        """
        if self.isfc_matrix is None:
            raise ValueError("Must compute ISFC first with compute_isfc()")
        
        print("\nSaving ISFC results to BIDS derivatives...")
        
        save_isfc_to_bids(
            isfc_matrix=self.isfc_matrix,
            subject_pairs=self.subject_pairs,
            roi_pair_names=self.roi_pair_names,
            bids_root=self.bids_root,
            session=self._session,
            task=self._task,
            run=self._run,
            space=self._space,
            seg=self._seg,
        )
        
        print("✓ Results saved!")
        
        return self
    
    def run(
        self,
        subjects: Optional[List[str]] = None,
        roi_indices: Optional[Union[List[int], np.ndarray]] = None,
        roi_names: Optional[List[str]] = None,
        symmetric: bool = True,
        save: bool = True,
    ):
        """
        Run complete ISFC pipeline.
        
        Parameters
        ----------
        subjects : list of str, optional
            List of subject IDs to load
        roi_indices : list or array, optional
            Indices of ROIs to use
        roi_names : list of str, optional
            Names of ROIs to use (overrides roi_indices)
        symmetric : bool, default: True
            Compute symmetric ISFC
        save : bool, default: True
            Save results to BIDS derivatives
        
        Returns
        -------
        self : ISFCPipeline
            Pipeline object with computed results
        """
        self.load_data(subjects=subjects)
        self.select_rois(roi_indices=roi_indices, roi_names=roi_names)
        self.compute_isfc(symmetric=symmetric)
        self.print_summary()
        
        if save:
            self.save_results()
        
        return self
    
    def run_group_analysis(
        self,
        participants_df: Optional[pd.DataFrame] = None,
        group_column: str = "group",
        behavior_column: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run group-level analyses on computed ISFC data.
        
        Performs three analyses: group comparisons, brain-behavior correlations,
        and within/between-group ISFC comparisons.
        
        Parameters
        ----------
        participants_df : DataFrame, optional
            Participant demographics and behavioral data. If None, loads from
            participants.tsv in BIDS root.
        group_column : str, default: "group"
            Column name for group labels
        behavior_column : str, optional
            Column name for behavioral measure. If provided, computes
            brain-behavior correlations.
        output_dir : str or Path, optional
            Directory to save results. If None, uses
            derivatives/pynatfmri_group_analysis/
        
        Returns
        -------
        results : dict
            Dictionary with keys:
            - 'group_comparison': Group comparison statistics
            - 'brain_behavior': Brain-behavior correlations (if behavior_column provided)
            - 'within_between': Within-group vs between-group comparisons
        """
        from .group_analysis import (
            load_participants_data,
            load_isfc_results,
            compare_groups,
            brain_behavior_correlation,
            within_vs_between_group_isfc,
            save_group_analysis_results,
        )
        
        print("\n" + "=" * 70)
        print("Running Group-Level Analyses")
        print("=" * 70)
        
        # Load participant data if not provided
        if participants_df is None:
            print(f"\nLoading participant data from {self.bids_root}...")
            participants_df = load_participants_data(self.bids_root)
        
        # Load ISFC results from saved files
        print("\nLoading ISFC results from derivatives...")
        isfc_data, roi_pairs = load_isfc_results(
            bids_root=self.bids_root,
            session=self._session,
            task=self._task,
            run=self._run,
            space=self._space,
            seg=self._seg,
        )
        
        # Set up output directory
        if output_dir is None:
            output_dir = self.bids_root / "derivatives" / "pynatfmri_group_analysis"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Analysis 1: Group comparisons
        print(f"\nComparing ISFC between {group_column} groups...")
        group_comparison = compare_groups(
            isfc_data=isfc_data,
            participants_df=participants_df,
            group_column=group_column,
        )
        
        sig_results = group_comparison[group_comparison['p_value_fdr'] < 0.05]
        print(f"✓ {len(sig_results)} ROI pairs with significant group differences (FDR < 0.05)")
        
        save_group_analysis_results(
            group_comparison,
            output_dir / "group_comparison.tsv",
            "Group comparison results"
        )
        results['group_comparison'] = group_comparison
        
        # Analysis 2: Brain-behavior correlations (if behavioral measure provided)
        if behavior_column is not None:
            print(f"\nComputing brain-behavior correlations with {behavior_column}...")
            brain_behavior = brain_behavior_correlation(
                isfc_data=isfc_data,
                participants_df=participants_df,
                behavior_column=behavior_column,
                group_column=group_column,
            )
            
            for group_name in brain_behavior['group'].unique():
                group_data = brain_behavior[brain_behavior['group'] == group_name]
                sig_corr = group_data[group_data['p_value_fdr'] < 0.05]
                print(f"  {group_name}: {len(sig_corr)} significant correlations (FDR < 0.05)")
            
            save_group_analysis_results(
                brain_behavior,
                output_dir / "brain_behavior_correlation.tsv",
                "Brain-behavior correlation results"
            )
            results['brain_behavior'] = brain_behavior
        
        # Analysis 3: Within-group vs between-group
        print(f"\nComparing within-group vs between-group ISFC...")
        within_between = within_vs_between_group_isfc(
            isfc_data=isfc_data,
            participants_df=participants_df,
            group_column=group_column,
        )
        
        save_group_analysis_results(
            within_between,
            output_dir / "within_vs_between_group.tsv",
            "Within-group vs between-group results"
        )
        results['within_between'] = within_between
        
        print("\n" + "=" * 70)
        print(f"Group analysis complete! Results saved to: {output_dir}")
        print("=" * 70)
        
        return results


def run_isfc_analysis(
    bids_root: Union[str, Path],
    derivative_name: str = "xcp_d",
    subjects: Optional[List[str]] = None,
    roi_indices: Optional[Union[List[int], np.ndarray]] = None,
    roi_names: Optional[List[str]] = None,
    session: str = "01",
    task: str = "rest",
    run: str = "01",
    space: str = "MNI152NLin6Asym",
    seg: str = "schaefer200",
    symmetric: bool = True,
    save: bool = True,
) -> ISFCPipeline:
    """
    Run complete ISFC analysis pipeline.
    
    Convenience function for running ISFC analysis in one step.
    
    Parameters
    ----------
    bids_root : str or Path
        Path to BIDS root directory
    derivative_name : str, default: "xcp_d"
        Name of derivatives folder
    subjects : list of str, optional
        Subject IDs to include
    roi_indices : list or array, optional
        Indices of ROIs to analyze
    roi_names : list of str, optional
        Names of ROIs to analyze
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
    symmetric : bool, default: True
        Compute symmetric ISFC
    save : bool, default: True
        Save results
    
    Returns
    -------
    pipeline : ISFCPipeline
        Pipeline object with results
    
    Examples
    --------
    >>> from pynatfmri.pipeline import run_isfc_analysis
    >>> pipeline = run_isfc_analysis(
    ...     bids_root="path/to/bids",
    ...     derivative_name="xcp_d",
    ...     roi_names=["roi_001", "roi_002", "roi_003"],
    ... )
    """
    pipeline = ISFCPipeline(
        bids_root=bids_root,
        derivative_name=derivative_name,
        session=session,
        task=task,
        run=run,
        space=space,
        seg=seg,
    )
    
    return pipeline.run(
        subjects=subjects,
        roi_indices=roi_indices,
        roi_names=roi_names,
        symmetric=symmetric,
        save=save,
    )

def run_group_analysis(
    bids_root: Union[str, Path],
    participants_df: Optional[pd.DataFrame] = None,
    group_column: str = "group",
    behavior_column: Optional[str] = None,
    session: str = "01",
    task: str = "rest",
    run: str = "01",
    space: str = "MNI152NLin6Asym",
    seg: str = "schaefer200",
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run complete group-level analysis pipeline.
    
    Convenience function for running group analyses in one step.
    Performs group comparisons, brain-behavior correlations, and
    within/between-group ISFC comparisons.
    
    Parameters
    ----------
    bids_root : str or Path
        Path to BIDS root directory
    participants_df : DataFrame, optional
        Participant demographics and behavioral data. If None, loads from
        participants.tsv in BIDS root.
    group_column : str, default: "group"
        Column name for group labels
    behavior_column : str, optional
        Column name for behavioral measure. If provided, computes
        brain-behavior correlations.
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
    output_dir : str or Path, optional
        Directory to save results. If None, uses
        derivatives/pynatfmri_group_analysis/
    
    Returns
    -------
    results : dict
        Dictionary with group analysis results:
        - 'group_comparison': Group comparison statistics
        - 'brain_behavior': Brain-behavior correlations (if behavior_column provided)
        - 'within_between': Within-group vs between-group comparisons
    
    Examples
    --------
    >>> from pynatfmri.pipeline import run_group_analysis
    >>> results = run_group_analysis(
    ...     bids_root="path/to/bids",
    ...     group_column="group",
    ...     behavior_column="mean_cued",
    ... )
    >>> print(results['group_comparison'].head())
    """
    pipeline = ISFCPipeline(
        bids_root=bids_root,
        session=session,
        task=task,
        run=run,
        space=space,
        seg=seg,
    )
    
    return pipeline.run_group_analysis(
        participants_df=participants_df,
        group_column=group_column,
        behavior_column=behavior_column,
        output_dir=output_dir,
    )