"""
Classifier Utilities Module

分類器関連のユーティリティ関数群
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings


def filter_prediction_points(
    features: np.ndarray,
    predictions: np.ndarray,
    tda_scores: Optional[np.ndarray] = None,
    pd_threshold: float = 0.1,
    density_threshold: float = 10,
    coordinate_threshold: float = 0.0
) -> List[Tuple[int, int, float]]:
    """
    予測ポイントのフィルタリング
    
    Parameters:
    -----------
    features : np.ndarray
        特徴量（座標など）
    predictions : np.ndarray
        予測ラベル (0:class_a, 1:class_b, 2:neutral)
    tda_scores : np.ndarray, optional
        TDAスコア（持続性など）
    pd_threshold : float
        持続性の閾値
    density_threshold : float
        密度の閾値
    coordinate_threshold : float
        座標の閾値
        
    Returns:
    --------
    filtered_points : List[Tuple[int, int, float]]
        フィルタ済みポイント (インデックス, ラベル, スコア)
    """
    filtered_points = []
    
    for i, pred in enumerate(predictions):
        include_point = False
        score = 0.0
        
        if pred == 2:  # neutralの場合
            # TDAスコアでフィルタリング
            if tda_scores is not None:
                if len(tda_scores.shape) > 1:
                    # 複数のTDA特徴がある場合（例：PD+simplex_tree）
                    pd_score = tda_scores[i, 0] if tda_scores.shape[1] > 0 else 0
                    density_score = tda_scores[i, 2] if tda_scores.shape[1] > 2 else 0  # num_edges
                else:
                    pd_score = tda_scores[i]
                    density_score = 0
                
                # 低持続性または低密度をノイズとして再分類
                if pd_score < pd_threshold or density_score < density_threshold:
                    # 座標で class_a/class_b を判定
                    if len(features.shape) > 1 and features.shape[1] > 0:
                        pred = 0 if features[i, 0] < coordinate_threshold else 1
                    else:
                        pred = 0  # デフォルト
                    include_point = True
                    score = pd_score
            else:
                # TDAスコアがない場合はそのまま
                include_point = True
                score = 1.0
        
        elif pred in [0, 1]:  # class_a/class_bの場合
            include_point = True
            if tda_scores is not None:
                score = tda_scores[i, 0] if len(tda_scores.shape) > 1 else tda_scores[i]
            else:
                score = 1.0
        
        if include_point:
            filtered_points.append((i, int(pred), float(score)))
    
    return filtered_points


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    分類性能の評価
    
    Parameters:
    -----------
    y_true : np.ndarray
        真のラベル
    y_pred : np.ndarray
        予測ラベル
    labels : List[str], optional
        ラベル名のリスト
    verbose : bool
        詳細な結果を表示するかどうか
        
    Returns:
    --------
    metrics : Dict[str, Any]
        評価指標の辞書
    """
    if labels is None:
        labels = ['class_a', 'class_b', 'neutral']
    
    # 基本的な評価指標
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # クラス別の評価
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }
    
    if verbose:
        print("=== 分類性能評価 ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1-score (weighted): {f1:.4f}")
        print()
        
        print("クラス別評価:")
        for i, label in enumerate(labels):
            if i < len(precision_per_class):
                print(f"{label:6s} - Precision: {precision_per_class[i]:.4f}, "
                      f"Recall: {recall_per_class[i]:.4f}, "
                      f"F1: {f1_per_class[i]:.4f}")
        print()
        
        # 詳細レポート
        print("詳細レポート:")
        try:
            report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
            print(report)
        except Exception as e:
            print(f"レポート生成エラー: {e}")
    
    return metrics


def create_classification_signals(
    filtered_points: List[Tuple[int, int, float]],
    confidence_threshold: float = 0.5,
    min_score_threshold: float = 0.1
) -> Dict[str, List[int]]:
    """
    分類シグナルの生成
    
    Parameters:
    -----------
    filtered_points : List[Tuple[int, int, float]]
        フィルタ済みポイント (インデックス, ラベル, スコア)
    confidence_threshold : float
        信頼度の閾値
    min_score_threshold : float
        最小スコアの閾値
        
    Returns:
    --------
    signals : Dict[str, List[int]]
        分類シグナル {'class_a': [indices], 'class_b': [indices], 'neutral': [indices]}
    """
    signals = {'class_a': [], 'class_b': [], 'neutral': []}
    
    for idx, label, score in filtered_points:
        # スコアによるフィルタリング
        if score >= min_score_threshold:
            if label == 0:  # class_a
                signals['class_a'].append(idx)
            elif label == 1:  # class_b
                signals['class_b'].append(idx)
            elif label == 2:  # neutral
                signals['neutral'].append(idx)
    
    return signals


def calculate_sequence_metrics(
    signals: Dict[str, List[int]],
    values: np.ndarray,
    processing_cost: float = 0.001
) -> Dict[str, float]:
    """
    系列データの指標計算
    
    Parameters:
    -----------
    signals : Dict[str, List[int]]
        分類シグナル
    values : np.ndarray
        値データ
    processing_cost : float
        処理コスト
        
    Returns:
    --------
    metrics : Dict[str, float]
        系列指標
    """
    if len(values) == 0:
        return {'total_change': 0.0, 'num_operations': 0, 'success_rate': 0.0}
    
    # 操作の実行をシミュレーション
    operations = []
    current_state = None
    
    # 全シグナルを時系列順にソート
    all_signals = []
    for signal_type, indices in signals.items():
        for idx in indices:
            all_signals.append((idx, signal_type))
    
    all_signals.sort(key=lambda x: x[0])
    
    for idx, signal_type in all_signals:
        if idx >= len(values):
            continue
            
        current_value = values[idx]
        
        if signal_type == 'class_b' and current_state is None:
            # 状態B開始
            current_state = {
                'type': 'state_b',
                'entry_value': current_value,
                'entry_idx': idx
            }
        elif signal_type == 'class_a' and current_state is None:
            # 状態A開始
            current_state = {
                'type': 'state_a',
                'entry_value': current_value,
                'entry_idx': idx
            }
        elif current_state is not None:
            # 状態終了
            if ((current_state['type'] == 'state_b' and signal_type == 'class_a') or
                (current_state['type'] == 'state_a' and signal_type == 'class_b')):
                
                entry_value = current_state['entry_value']
                
                if current_state['type'] == 'state_b':
                    raw_change = (current_value - entry_value) / entry_value
                else:  # state_a
                    raw_change = (entry_value - current_value) / entry_value
                
                # 処理コストを差し引く
                net_change = raw_change - 2 * processing_cost
                
                operations.append({
                    'type': current_state['type'],
                    'entry_value': entry_value,
                    'exit_value': current_value,
                    'entry_idx': current_state['entry_idx'],
                    'exit_idx': idx,
                    'change': net_change
                })
                
                current_state = None
    
    # 指標計算
    if len(operations) == 0:
        return {'total_change': 0.0, 'num_operations': 0, 'success_rate': 0.0}
    
    changes = [op['change'] for op in operations]
    total_change = np.sum(changes)
    num_operations = len(operations)
    successful_operations = sum(1 for c in changes if c > 0)
    success_rate = successful_operations / num_operations if num_operations > 0 else 0
    
    metrics = {
        'total_change': total_change,
        'average_change': np.mean(changes),
        'num_operations': num_operations,
        'success_rate': success_rate,
        'max_change': np.max(changes),
        'min_change': np.min(changes),
        'volatility': np.std(changes)
    }
    
    return metrics
