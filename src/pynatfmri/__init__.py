"""
pynatfmri: A Python library for neuroimaging fMRI connectivity analysis

This module provides tools for analyzing functional magnetic resonance imaging (fMRI) data,
with a focus on functional connectivity analysis.
"""

__version__ = "0.1.0"
__author__ = "J. Bartolotti"

# Import main modules
from . import isfc
from . import pipeline

# Import commonly used pipeline functions
from .pipeline import ISFCPipeline, run_isfc_analysis

__all__ = ["isfc", "pipeline", "ISFCPipeline", "run_isfc_analysis"]
