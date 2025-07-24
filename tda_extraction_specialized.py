"""
Specialized TDA Feature Extraction

PDスコアとsimplex_tree特徴抽出に特化したモジュール
遅延埋め込み対応・ウィンドウサイズ対応
"""

import numpy as np
import gudhi as gd
import networkx as nx
from typing import Tuple, List, Optional
import warnings


def time_delay_embedding_multivariate(
    time_series: np.ndarray,
    embedding_dim: int = 3,
    tau: int = 1,
    selected_features: Optional[List[int]] = None
) -> np.ndarray:
    """
    多変量時系列の遅延埋め込み
    
    Parameters:
    -----------
    time_series : np.ndarray
        時系列データ (window_size, n_features)
    embedding_dim : int
        埋め込み次元
    tau : int
        遅延ステップ
    selected_features : List[int], optional
        使用する特徴量のインデックス（Noneの場合は平均を使用）
        
    Returns:
    --------
    points : np.ndarray
        埋め込まれた点群 (n_points, embedding_dim * n_selected_features)
    """
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1, 1)
    
    window_size, n_features = time_series.shape
    
    if selected_features is None:
        # 全特徴量の平均を使用
        series = np.mean(time_series, axis=1)
        series = series.reshape(-1, 1)
    else:
        # 指定された特徴量のみ使用
        series = time_series[:, selected_features]
    
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    
    n_series = series.shape[1]
    
    # 点数計算
    n_points = window_size - (embedding_dim - 1) * tau
    
    if n_points <= 0:
        raise ValueError(f"時系列が短すぎます。長さ{window_size}、必要な長さ{(embedding_dim - 1) * tau + 1}")
    
    # 各系列に対して遅延埋め込みを実行
    embedded_series = []
    
    for j in range(n_series):
        single_series = series[:, j]
        embedded_single = np.array([
            single_series[i:i + embedding_dim * tau:tau] 
            for i in range(n_points)
        ])
        embedded_series.append(embedded_single)
    
    # 全系列を結合
    if len(embedded_series) == 1:
        points = embedded_series[0]
    else:
        points = np.hstack(embedded_series)
    
    return points


def compute_pd_and_simplex_features(
    time_series_windows: np.ndarray,
    embedding_dim: int = 3,
    tau: int = 1,
    max_edge_length: float = 2.0,
    max_dimension: int = 2,
    persistence_threshold: float = 0.1,
    selected_features: Optional[List[int]] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    複数ウィンドウからPDスコアとsimplex_tree特徴を計算
    
    Parameters:
    -----------
    time_series_windows : np.ndarray
        時系列ウィンドウ (n_windows, window_size, n_features)
    embedding_dim : int
        遅延埋め込み次元
    tau : int
        遅延ステップ
    max_edge_length : float
        Rips複体の最大エッジ長
    max_dimension : int
        計算する最大次元
    persistence_threshold : float
        持続性フィルタの閾値
    selected_features : List[int], optional
        使用する特徴量のインデックス
    verbose : bool
        進捗表示
        
    Returns:
    --------
    pd_scores : np.ndarray
        PDスコア (n_windows, 1)
    simplex_features : np.ndarray
        simplex_tree特徴 (n_windows, 11)
    """
    if time_series_windows.ndim == 2:
        # 単一ウィンドウの場合
        time_series_windows = time_series_windows[np.newaxis, ...]
    
    n_windows = time_series_windows.shape[0]
    
    pd_scores = []
    simplex_features = []
    
    for i, window in enumerate(time_series_windows):
        if verbose and i % 10 == 0:
            print(f"Processing window {i+1}/{n_windows}")
        
        try:
            # 1. 遅延埋め込み
            points = time_delay_embedding_multivariate(
                window, 
                embedding_dim=embedding_dim,
                tau=tau,
                selected_features=selected_features
            )
            
            # 2. Rips複体とsimplex_tree生成
            rips = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
            simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
            
            # 3. 持続ホモロジー計算
            persistence = simplex_tree.persistence()
            
            # 4. PDスコア抽出
            pd_score = extract_pd_score_detailed(persistence, persistence_threshold)
            pd_scores.append(pd_score)
            
            # 5. simplex_tree特徴抽出
            st_features = extract_simplex_tree_features_detailed(simplex_tree)
            simplex_features.append(st_features)
            
        except Exception as e:
            if verbose:
                warnings.warn(f"Window {i}でエラー: {e}")
            
            # エラー時はデフォルト値
            pd_scores.append(0.0)
            simplex_features.append(np.zeros(11))
    
    pd_scores = np.array(pd_scores).reshape(-1, 1)
    simplex_features = np.array(simplex_features)
    
    if verbose:
        print(f"処理完了: {n_windows} windows")
        print(f"PDスコア統計: mean={pd_scores.mean():.4f}, std={pd_scores.std():.4f}")
        print(f"Simplex特徴統計: mean={simplex_features.mean():.4f}")
    
    return pd_scores, simplex_features


def extract_pd_score_detailed(
    persistence: List,
    threshold: float = 0.1
) -> float:
    """
    詳細なPDスコア抽出（複数次元対応）
    
    Parameters:
    -----------
    persistence : List
        持続ホモロジーの結果
    threshold : float
        持続性フィルタの閾値
        
    Returns:
    --------
    pd_score : float
        統合されたPDスコア
    """
    scores_by_dim = {}
    
    for _, (dim, (birth, death)) in persistence:
        if death == float('inf'):
            continue
        
        duration = death - birth
        if duration > threshold:
            if dim not in scores_by_dim:
                scores_by_dim[dim] = []
            scores_by_dim[dim].append(duration)
    
    # 各次元の平均持続時間を計算
    dim_scores = []
    for dim in [0, 1, 2]:  # 0, 1, 2次元
        if dim in scores_by_dim and len(scores_by_dim[dim]) > 0:
            dim_score = np.mean(scores_by_dim[dim])
        else:
            dim_score = 0.0
        dim_scores.append(dim_score)
    
    # 重み付き平均（1次元を重視）
    weights = [0.2, 0.6, 0.2]  # 0次元, 1次元, 2次元の重み
    pd_score = np.average(dim_scores, weights=weights)
    
    return pd_score


def extract_simplex_tree_features_detailed(
    simplex_tree
) -> np.ndarray:
    """
    詳細なsimplex_tree特徴抽出
    
    Parameters:
    -----------
    simplex_tree : gudhi.SimplexTree
        シンプレックスツリー
        
    Returns:
    --------
    features : np.ndarray
        特徴ベクトル (11次元)
    """
    # NetworkXグラフ生成（1次元スケルトン）
    G = nx.Graph()
    for simplex, _ in simplex_tree.get_skeleton(1):
        if len(simplex) == 1:
            G.add_node(simplex[0])
        elif len(simplex) == 2:
            G.add_edge(simplex[0], simplex[1])
    
    # 基本統計
    num_vertices = simplex_tree.num_vertices()
    num_edges = G.number_of_edges()
    
    # Betti数
    betti_numbers = simplex_tree.betti_numbers()
    betti_0 = betti_numbers[0] if len(betti_numbers) > 0 else 0
    betti_1 = betti_numbers[1] if len(betti_numbers) > 1 else 0
    
    # フィルトレーション値統計
    filtration_values = [f for _, f in simplex_tree.get_filtration()]
    if len(filtration_values) == 0:
        filtration_values = [0.0]
    
    avg_filtration = np.mean(filtration_values)
    filtration_var = np.var(filtration_values)
    max_filtration = np.max(filtration_values)
    
    # グラフ統計
    if num_vertices > 0 and G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if len(degrees) > 0 else 0
        avg_degree = np.mean(list(degrees.values())) if len(degrees) > 0 else 0
        
        # クラスタリング係数
        try:
            clustering_coeff = nx.average_clustering(G)
        except:
            clustering_coeff = 0.0
    else:
        max_degree = 0
        avg_degree = 0
        clustering_coeff = 0.0
    
    # 高次simplex統計
    num_triangles = 0
    try:
        for simplex, _ in simplex_tree.get_skeleton(2):
            if len(simplex) == 3:  # 三角形
                num_triangles += 1
    except:
        num_triangles = 0
    
    # 特徴ベクトル
    features = np.array([
        num_vertices,      # 0: ノード数
        num_edges,         # 1: エッジ数  
        betti_0,           # 2: 連結成分数
        betti_1,           # 3: 1次元穴の数
        avg_filtration,    # 4: 平均フィルトレーション
        max_degree,        # 5: 最大次数
        avg_degree,        # 6: 平均次数
        clustering_coeff,  # 7: クラスタリング係数
        num_triangles,     # 8: 三角形数
        filtration_var,    # 9: フィルトレーション分散
        max_filtration     # 10: 最大フィルトレーション
    ])
    
    return features


def batch_process_windows(
    time_series_data: np.ndarray,
    window_size: int = 40,
    step_size: int = 1,
    embedding_dim: int = 3,
    tau: int = 1,
    max_edge_length: float = 2.0,
    max_dimension: int = 2,
    persistence_threshold: float = 0.1,
    selected_features: Optional[List[int]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    完全なバッチ処理パイプライン
    
    Parameters:
    -----------
    time_series_data : np.ndarray
        時系列データ (n_samples, n_features)
    window_size : int
        ウィンドウサイズ
    step_size : int
        ステップサイズ
    embedding_dim : int
        遅延埋め込み次元
    tau : int
        遅延ステップ
    max_edge_length : float
        Rips複体の最大エッジ長
    max_dimension : int
        計算する最大次元
    persistence_threshold : float
        持続性フィルタの閾値
    selected_features : List[int], optional
        使用する特徴量のインデックス
    verbose : bool
        進捗表示
        
    Returns:
    --------
    pd_scores : np.ndarray
        PDスコア (n_windows, 1)
    simplex_features : np.ndarray
        simplex_tree特徴 (n_windows, 11)
    """
    if verbose:
        print("=== TDA特徴抽出バッチ処理開始 ===")
        print(f"入力データ形状: {time_series_data.shape}")
        print(f"ウィンドウサイズ: {window_size}")
        print(f"遅延埋め込み: dim={embedding_dim}, tau={tau}")
    
    # 1. ウィンドウ作成
    n_samples, n_features = time_series_data.shape
    n_windows = (n_samples - window_size) // step_size + 1
    
    windows = []
    for i in range(0, n_windows * step_size, step_size):
        if i + window_size <= n_samples:
            window = time_series_data[i:i + window_size]
            windows.append(window)
    
    windows = np.array(windows)
    
    if verbose:
        print(f"作成されたウィンドウ数: {len(windows)}")
    
    # 2. TDA特徴抽出
    pd_scores, simplex_features = compute_pd_and_simplex_features(
        windows,
        embedding_dim=embedding_dim,
        tau=tau,
        max_edge_length=max_edge_length,
        max_dimension=max_dimension,
        persistence_threshold=persistence_threshold,
        selected_features=selected_features,
        verbose=verbose
    )
    
    if verbose:
        print("=== TDA特徴抽出バッチ処理完了 ===")
    
    return pd_scores, simplex_features


# 使用例とテスト関数
if __name__ == "__main__":
    print("=== TDA特徴抽出テスト ===")
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 200
    n_features = 6
    
    # 模擬時系列データ
    t = np.linspace(0, 4, n_samples)
    data = np.zeros((n_samples, n_features))
    
    # 主要な値
    data[:, 0] = 100 + 10 * np.sin(t) + np.random.randn(n_samples) * 0.5
    # その他の特徴量
    for i in range(1, n_features):
        data[:, i] = data[:, 0] * (0.9 + 0.2 * np.random.randn(n_samples))
    
    print(f"サンプルデータ形状: {data.shape}")
    
    # TDA特徴抽出
    try:
        pd_scores, simplex_features = batch_process_windows(
            data,
            window_size=40,
            step_size=5,  # 高速化のため
            embedding_dim=3,
            tau=1,
            selected_features=[0, 5],  # Close価格とRSI相当
            verbose=True
        )
        
        print(f"\n結果:")
        print(f"PDスコア形状: {pd_scores.shape}")
        print(f"Simplex特徴形状: {simplex_features.shape}")
        print(f"PDスコア統計: min={pd_scores.min():.4f}, max={pd_scores.max():.4f}, mean={pd_scores.mean():.4f}")
        print(f"特徴名:")
        feature_names = [
            'num_vertices', 'num_edges', 'betti_0', 'betti_1', 'avg_filtration',
            'max_degree', 'avg_degree', 'clustering_coeff', 'num_triangles',
            'filtration_var', 'max_filtration'
        ]
        for i, name in enumerate(feature_names):
            print(f"  {i}: {name} (mean: {simplex_features[:, i].mean():.4f})")
        
    except Exception as e:
        print(f"エラー: {e}")
        print("必要なライブラリ（gudhi, networkx）がインストールされていない可能性があります。")
