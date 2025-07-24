"""
Persistent Homology Classifier Package

持続ホモロジーを使った時系列データの分類・パターン認識パッケージ
TDA（トポロジカルデータ解析）による特徴抽出と分類を提供
"""

from .tda_features import (
    TDAFeatureExtractor,
    compute_persistence_diagram,
    extract_pd_scores,
    extract_simplex_tree_features,
    time_delay_embedding
)

from .data_preprocessing import (
    signed_log_transform,
    create_sliding_windows,
    normalize_data
)

from .classifier_utils import (
    filter_prediction_points,
    evaluate_classification
)

__version__ = "1.0.0"
__author__ = "Persistent Homology Team"

__all__ = [
    'TDAFeatureExtractor',
    'compute_persistence_diagram',
    'extract_pd_scores', 
    'extract_simplex_tree_features',
    'time_delay_embedding',
    'signed_log_transform',
    'create_sliding_windows',
    'normalize_data',
    'filter_prediction_points',
    'evaluate_classification'
]
