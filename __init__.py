"""
Hybrid Clearance and Obstacle Detection Solution for Metro Moscow Autonomous Train (Project 14).
Combines:
- Tier 1: Curvilinear clearance envelope (ГОСТ 9238) + Bore-Clearance Monitor (45-200m).
- Tier 2: Lightweight invariant geometry shape descriptors + One-Class Mahalanobis anomaly classifier.
"""

from hybrid_clearance_solution.tier1_curvilinear_detector import Tier1CurvilinearDetector
from hybrid_clearance_solution.tier2_feature_extractor import extract_cluster_features, FEATURE_NAMES
from hybrid_clearance_solution.tier2_classifier import Tier2Classifier
from hybrid_clearance_solution.hybrid_pipeline import HybridPerceptionPipeline

__all__ = [
    "Tier1CurvilinearDetector",
    "extract_cluster_features",
    "FEATURE_NAMES",
    "Tier2Classifier",
    "HybridPerceptionPipeline",
]
