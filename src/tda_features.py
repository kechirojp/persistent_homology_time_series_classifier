"""
TDA Features Module

持続ホモロジーとシンプレックスツリーから特徴量を抽出するモジュール
"""

import numpy as np
import gudhi as gd
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union
import warnings

class TDAFeatureExtractor:
    """
    TDA（位相的データ解析）特徴抽出器
    
    遅延埋め込み、持続ホモロジー計算、シンプレックスツリー特徴抽出を統合
    """
    
    def __init__(
        self,
        embedding_dim: int = 3,
        tau: int = 1,
        max_edge_length: float = 2.0,
        max_dimension: int = 2,
        persistence_threshold: float = 0.1
    ):
        """
        初期化
        
        Parameters:
        -----------
        embedding_dim : int
            遅延埋め込みの次元数（デフォルト: 3）
        tau : int  
            遅延ステップ（デフォルト: 1）
        max_edge_length : float
            Rips複体の最大エッジ長（デフォルト: 2.0）
        max_dimension : int
            計算する最大次元（デフォルト: 2）
        persistence_threshold : float
            持続性フィルタの閾値（デフォルト: 0.1）
        """
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        self.persistence_threshold = persistence_threshold
    
    def extract_features(
        self, 
        time_series: np.ndarray,
        return_components: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        時系列データからTDA特徴を抽出
        
        Parameters:
        -----------
        time_series : np.ndarray
            時系列データ (n_windows, window_size, n_features) または (window_size, n_features)
        return_components : bool
            中間結果（PD、simplex_tree特徴）も返すかどうか
            
        Returns:
        --------
        features : np.ndarray
            統合されたTDA特徴ベクトル
        components : Dict (return_components=Trueの場合)
            中間結果の辞書
        """
        if time_series.ndim == 2:
            # 単一ウィンドウの場合は3次元に拡張
            time_series = time_series[np.newaxis, ...]
        
        n_windows = time_series.shape[0]
        
        # 特徴抽出
        pd_scores = []
        st_features = []
        
        for i, window in enumerate(time_series):
            try:
                # 遅延埋め込み
                points = time_delay_embedding(
                    window, 
                    self.embedding_dim, 
                    self.tau
                )
                
                # 持続ホモロジー計算
                persistence = compute_persistence_diagram(
                    points,
                    self.max_edge_length,
                    self.max_dimension
                )
                
                # PDスコア抽出
                pd_score = extract_pd_scores(
                    persistence,
                    self.persistence_threshold
                )
                pd_scores.append(pd_score)
                
                # シンプレックスツリー特徴抽出
                st_feature = extract_simplex_tree_features(
                    points,
                    self.max_edge_length,
                    self.max_dimension
                )
                st_features.append(st_feature)
                
            except Exception as e:
                warnings.warn(f"Window {i}でエラー発生: {e}")
                # デフォルト値で補填
                pd_scores.append(0.0)
                st_features.append(np.zeros(11))  # デフォルト特徴数
        
        # 配列化
        pd_scores = np.array(pd_scores).reshape(-1, 1)
        st_features = np.array(st_features)
        
        # 統合特徴
        integrated_features = np.hstack([pd_scores, st_features])
        
        if return_components:
            components = {
                'pd_scores': pd_scores,
                'simplex_tree_features': st_features
            }
            return integrated_features, components
        
        return integrated_features


def time_delay_embedding(
    time_series: np.ndarray,
    embedding_dim: int = 3,
    tau: int = 1
) -> np.ndarray:
    """
    遅延埋め込みによる点群生成
    
    Parameters:
    -----------
    time_series : np.ndarray
        時系列データ (window_size, n_features)
    embedding_dim : int
        埋め込み次元
    tau : int
        遅延ステップ
        
    Returns:
    --------
    points : np.ndarray
        埋め込まれた点群 (n_points, embedding_dim)
    """
    if time_series.ndim == 2:
        # 多変量の場合は平均系列を作成（簡易版）
        series = np.mean(time_series, axis=1)
    else:
        series = time_series
    
    # 点数計算
    n_points = len(series) - (embedding_dim - 1) * tau
    
    if n_points <= 0:
        raise ValueError(f"時系列が短すぎます。長さ{len(series)}、必要な長さ{(embedding_dim - 1) * tau + 1}")
    
    # 遅延埋め込み
    points = np.array([
        series[i:i + embedding_dim * tau:tau] 
        for i in range(n_points)
    ])
    
    return points


def compute_persistence_diagram(
    points: np.ndarray,
    max_edge_length: float = 2.0,
    max_dimension: int = 2
) -> List:
    """
    点群から持続ホモロジーを計算
    
    Parameters:
    -----------
    points : np.ndarray
        点群データ (n_points, dim)
    max_edge_length : float
        Rips複体の最大エッジ長
    max_dimension : int
        計算する最大次元
        
    Returns:
    --------
    persistence : List
        持続ホモロジーの結果
    """
    # Rips複体生成
    rips = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
    
    # 持続ホモロジー計算
    persistence = simplex_tree.persistence()
    
    return persistence


def extract_pd_scores(
    persistence: List,
    threshold: float = 0.1
) -> float:
    """
    持続ホモロジーからPDスコアを抽出
    
    Parameters:
    -----------
    persistence : List
        持続ホモロジーの結果
    threshold : float
        持続性フィルタの閾値
        
    Returns:
    --------
    pd_score : float
        PDスコア（平均持続時間）
    """
    # 1次元の持続ペアを抽出してフィルタリング
    durations = np.array([
        d - b for _, (dim, (b, d)) in persistence 
        if dim == 1 and (d - b) > threshold and d != float('inf')
    ])
    
    # 平均持続時間を計算
    pd_score = durations.mean() if len(durations) > 0 else 0.0
    
    return pd_score


def extract_simplex_tree_features(
    points: np.ndarray,
    max_edge_length: float = 2.0,
    max_dimension: int = 2
) -> np.ndarray:
    """
    シンプレックスツリーから統計的特徴を抽出
    
    Parameters:
    -----------
    points : np.ndarray
        点群データ (n_points, dim)
    max_edge_length : float
        Rips複体の最大エッジ長
    max_dimension : int
        計算する最大次元
        
    Returns:
    --------
    features : np.ndarray
        シンプレックスツリー特徴ベクトル (11次元)
    """
    # Rips複体とシンプレックスツリー生成
    rips = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
    
    # NetworkXグラフに変換（1次元スケルトン）
    G = nx.Graph()
    for simplex, _ in simplex_tree.get_skeleton(1):
        if len(simplex) == 1:
            G.add_node(simplex[0])
        elif len(simplex) == 2:
            G.add_edge(simplex[0], simplex[1])
    
    # 基本統計量
    num_nodes = simplex_tree.num_vertices()
    num_edges = G.number_of_edges()
    
    # Betti数
    betti_numbers = simplex_tree.betti_numbers()
    betti_0 = betti_numbers[0] if len(betti_numbers) > 0 else 0
    betti_1 = betti_numbers[1] if len(betti_numbers) > 1 else 0
    
    # フィルトレーション値
    filtration_values = [f for _, f in simplex_tree.get_filtration()]
    if len(filtration_values) == 0:
        filtration_values = [0.0]
    
    avg_filtration = np.mean(filtration_values)
    filtration_var = np.var(filtration_values)
    max_filtration = np.max(filtration_values)
    
    # グラフ統計量
    if num_nodes > 0 and num_edges > 0:
        max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        clustering_coeff = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
    else:
        max_degree = 0
        avg_degree = 0
        clustering_coeff = 0
    
    # 三角形数（2-simplex）
    num_triangles = sum(1 for _ in simplex_tree.get_skeleton(2)) - num_edges - num_nodes
    num_triangles = max(0, num_triangles)  # 負の値を防ぐ
    
    # 特徴ベクトル組み立て
    features = np.array([
        num_nodes, num_edges, betti_0, betti_1, avg_filtration,
        max_degree, avg_degree, clustering_coeff, num_triangles,
        filtration_var, max_filtration
    ])
    
    return features
