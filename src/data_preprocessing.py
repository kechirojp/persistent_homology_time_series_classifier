"""
Data Preprocessing Module

データ前処理関連の関数群
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings

def signed_log_transform(
    data: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    正負符号付き対数変換
    
    Parameters:
    -----------
    data : np.ndarray
        入力データ
    epsilon : float
        ゼロ除算防止用の小さな値
        
    Returns:
    --------
    transformed_data : np.ndarray
        変換済みデータ
    """
    # 符号を保持しながら対数変換
    sign = np.sign(data)
    abs_data = np.abs(data)
    
    # ゼロを epsilon で置き換え
    abs_data = np.where(abs_data == 0, epsilon, abs_data)
    
    # 符号付き対数変換
    transformed = sign * np.log(abs_data + epsilon)
    
    return transformed


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    step_size: int = 1,
    labels: Optional[np.ndarray] = None
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    スライディングウィンドウでデータを分割
    
    Parameters:
    -----------
    data : np.ndarray
        時系列データ (n_samples, n_features)
    window_size : int
        ウィンドウサイズ
    step_size : int
        ステップサイズ（デフォルト: 1）
    labels : np.ndarray, optional
        ラベルデータ (n_samples,)
        
    Returns:
    --------
    windows : np.ndarray
        ウィンドウ化されたデータ (n_windows, window_size, n_features)
    window_labels : np.ndarray (labelsが提供された場合)
        各ウィンドウに対応するラベル
    """
    if len(data) < window_size:
        raise ValueError(f"データ長{len(data)}がウィンドウサイズ{window_size}より小さいです")
    
    # ウィンドウ数計算
    n_windows = (len(data) - window_size) // step_size + 1
    
    # ウィンドウ作成
    windows = []
    window_labels = []
    
    for i in range(0, n_windows * step_size, step_size):
        if i + window_size <= len(data):
            window = data[i:i + window_size]
            windows.append(window)
            
            if labels is not None:
                # ウィンドウの最後のラベルを使用（異常検知用）
                label = labels[i + window_size - 1]
                window_labels.append(label)
    
    windows = np.array(windows)
    
    if labels is not None:
        window_labels = np.array(window_labels)
        return windows, window_labels
    
    return windows


def normalize_data(
    data: np.ndarray,
    method: str = 'minmax',
    axis: int = 0,
    feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[np.ndarray, dict]:
    """
    データ正規化
    
    Parameters:
    -----------
    data : np.ndarray
        入力データ
    method : str
        正規化手法 ('minmax', 'zscore', 'robust')
    axis : int
        正規化する軸
    feature_range : Tuple[float, float]
        MinMax正規化の範囲
        
    Returns:
    --------
    normalized_data : np.ndarray
        正規化済みデータ
    scaler_params : dict
        正規化パラメータ（逆変換用）
    """
    if method == 'minmax':
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
        
        # ゼロ除算防止
        data_range = data_max - data_min
        data_range = np.where(data_range == 0, 1, data_range)
        
        # 正規化
        normalized = (data - data_min) / data_range
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        scaler_params = {
            'method': method,
            'data_min': data_min,
            'data_max': data_max,
            'feature_range': feature_range
        }
        
    elif method == 'zscore':
        data_mean = np.mean(data, axis=axis, keepdims=True)
        data_std = np.std(data, axis=axis, keepdims=True)
        
        # ゼロ除算防止
        data_std = np.where(data_std == 0, 1, data_std)
        
        normalized = (data - data_mean) / data_std
        
        scaler_params = {
            'method': method,
            'data_mean': data_mean,
            'data_std': data_std
        }
        
    elif method == 'robust':
        data_median = np.median(data, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        
        # ゼロ除算防止
        iqr = np.where(iqr == 0, 1, iqr)
        
        normalized = (data - data_median) / iqr
        
        scaler_params = {
            'method': method,
            'data_median': data_median,
            'iqr': iqr
        }
        
    else:
        raise ValueError(f"サポートされていない正規化手法: {method}")
    
    return normalized, scaler_params


def inverse_normalize(
    normalized_data: np.ndarray,
    scaler_params: dict
) -> np.ndarray:
    """
    正規化の逆変換
    
    Parameters:
    -----------
    normalized_data : np.ndarray
        正規化済みデータ
    scaler_params : dict
        正規化パラメータ
        
    Returns:
    --------
    original_data : np.ndarray
        元スケールのデータ
    """
    method = scaler_params['method']
    
    if method == 'minmax':
        data_min = scaler_params['data_min']
        data_max = scaler_params['data_max']
        feature_range = scaler_params['feature_range']
        
        # 逆変換
        data = (normalized_data - feature_range[0]) / (feature_range[1] - feature_range[0])
        data = data * (data_max - data_min) + data_min
        
    elif method == 'zscore':
        data_mean = scaler_params['data_mean']
        data_std = scaler_params['data_std']
        
        data = normalized_data * data_std + data_mean
        
    elif method == 'robust':
        data_median = scaler_params['data_median']
        iqr = scaler_params['iqr']
        
        data = normalized_data * iqr + data_median
        
    else:
        raise ValueError(f"サポートされていない正規化手法: {method}")
    
    return data


def handle_missing_values(
    data: np.ndarray,
    method: str = 'forward_fill',
    fill_value: float = 0.0
) -> np.ndarray:
    """
    欠損値処理
    
    Parameters:
    -----------
    data : np.ndarray
        入力データ
    method : str
        処理方法 ('forward_fill', 'backward_fill', 'interpolate', 'constant')
    fill_value : float
        定数埋めの値
        
    Returns:
    --------
    filled_data : np.ndarray
        欠損値処理済みデータ
    """
    data = data.copy()
    
    if method == 'forward_fill':
        # 前方埋め
        mask = np.isnan(data)
        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        data[mask] = data[idx[mask]]
        
    elif method == 'backward_fill':
        # 後方埋め
        data = data[::-1]  # 逆順
        mask = np.isnan(data)
        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        data[mask] = data[idx[mask]]
        data = data[::-1]  # 元に戻す
        
    elif method == 'interpolate':
        # 線形補間
        for col in range(data.shape[1] if data.ndim > 1 else 1):
            if data.ndim > 1:
                series = data[:, col]
            else:
                series = data
            
            mask = ~np.isnan(series)
            if mask.sum() > 1:  # 有効な値が2つ以上ある場合
                indices = np.arange(len(series))
                series[~mask] = np.interp(indices[~mask], indices[mask], series[mask])
            
            if data.ndim > 1:
                data[:, col] = series
            else:
                data = series
                
    elif method == 'constant':
        # 定数埋め
        data[np.isnan(data)] = fill_value
        
    else:
        raise ValueError(f"サポートされていない欠損値処理方法: {method}")
    
    return data
