"""Basic tests for pynatfmri"""

import pytest


def test_import():
    """Test that pynatfmri can be imported"""
    import pynatfmri
    assert pynatfmri.__version__ == "0.1.0"
