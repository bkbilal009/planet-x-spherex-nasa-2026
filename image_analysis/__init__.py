"""SPHEREx Image Analysis - multi-epoch change detection on real IRSA data.

Public entry points:
    nasa_data.search_observations / fetch_observation   archive access
    fits_processor.read_spherex_fits                    local FITS files
    pipeline.run_change_detection                       align + subtract + detect
"""
from .fits_processor import FitsReadError, Observation, read_spherex_fits
from .pipeline import ChangeDetectionResult, run_change_detection

__all__ = ["FitsReadError", "Observation", "read_spherex_fits",
           "ChangeDetectionResult", "run_change_detection"]
