"""
Example Usage of Persistent Homology Classifier

持続ホモロジー分類器の使用例（TDAのみ、UMAPなし）
"""

import numpy as np
import sys
import os

# パッケージをインポートパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import (
    TDAFeatureExtractor,
    signed_log_transform,
    create_sliding_windows,
    normalize_data,
    filter_prediction_points,
    evaluate_classification
)

from src.classifier_utils import (
    create_classification_signals,
    calculate_sequence_metrics
)

def generate_sample_data(n_samples: int = 1000, n_features: int = 6) -> tuple:
    """
    サンプルデータ生成（時系列データを模擬）
    
    【例え話：お店の来客データ】
    - n_samples：観測日数（例：1000日分）
    - 特徴量0-3：売上関連指標（メイン売上、関連商品売上など）
    - 特徴量4：ボリューム的特徴（来客数）
      → 好調/不調どちらでも注目度が上がると増加（絶対値使用）
    - 特徴量5：指標的特徴（顧客満足度）
      → 好調時は上昇、不調時は下降（tanh使用で-1〜+1範囲）
    
    Returns:
    --------
    data : np.ndarray
        模擬時系列データ (n_samples, n_features)
    labels : np.ndarray  
        ラベル (0:class_a, 1:class_b, 2:neutral)
    """
    np.random.seed(42)
    
    # 基本パターン生成
    t = np.linspace(0, 10, n_samples)
    pattern = np.sin(t) + 0.5 * np.sin(3*t) + 0.1 * np.random.randn(n_samples)
    
    # 各特徴量生成（多次元時系列）
    data = np.zeros((n_samples, n_features))
    
    # 主要な値（基準）- メイン売上
    data[:, 0] = 100 + 10 * pattern + np.random.randn(n_samples) * 0.5
    
    # 関連する特徴量 - 関連商品売上
    data[:, 1] = data[:, 0] + np.random.randn(n_samples) * 0.3  # feature_1
    data[:, 2] = data[:, 0] + np.abs(np.random.randn(n_samples)) * 0.5  # feature_2
    data[:, 3] = data[:, 0] - np.abs(np.random.randn(n_samples)) * 0.5  # feature_3
    
    # ボリューム的特徴 - 来客数（好調/不調どちらでも注目度で増加）
    data[:, 4] = 1000 + 200 * np.abs(pattern) + np.random.randn(n_samples) * 50
    
    # 指標的特徴 - 顧客満足度（好調時+、不調時-）
    data[:, 5] = 50 + 30 * np.tanh(pattern) + np.random.randn(n_samples) * 5
    
    # ラベル生成（パターンベース）
    labels = np.zeros(n_samples, dtype=int)
    
    # Class A（ピーク値）: パターンが高く、次に下がる
    class_a_condition = (pattern > 0.8) & (np.gradient(pattern) < -0.1)
    labels[class_a_condition] = 0
    
    # Class B（谷値）: パターンが低く、次に上がる  
    class_b_condition = (pattern < -0.8) & (np.gradient(pattern) > 0.1)
    labels[class_b_condition] = 1
    
    # Neutral（それ以外）
    labels[(~class_a_condition) & (~class_b_condition)] = 2
    
    return data, labels


def main():
    """メイン実行関数"""
    print("=== 持続ホモロジー分類器の使用例 ===\n")
    
    # 1. サンプルデータ生成
    print("1. サンプルデータ生成中...")
    data, labels = generate_sample_data(n_samples=500, n_features=6)
    print(f"データ形状: {data.shape}")
    print(f"ラベル分布: Class A={np.sum(labels==0)}, Class B={np.sum(labels==1)}, Neutral={np.sum(labels==2)}")
    
    # 2. データ前処理
    print("\n2. データ前処理中...")
    
    # 正負符号付き対数変換
    data_transformed = signed_log_transform(data)
    print(f"対数変換後形状: {data_transformed.shape}")
    
    # 正規化
    data_normalized, scaler_params = normalize_data(data_transformed, method='zscore')
    print(f"正規化後形状: {data_normalized.shape}")
    
    # 3. スライディングウィンドウ作成
    print("\n3. スライディングウィンドウ作成中...")
    windows, window_labels = create_sliding_windows(
        data_normalized, 
        window_size=40, 
        step_size=5,
        labels=labels
    )
    print(f"ウィンドウ形状: {windows.shape}")
    print(f"ウィンドウラベル形状: {window_labels.shape}")
    
    # 4. TDA特徴抽出
    print("\n4. TDA特徴抽出中...")
    
    # TDA特徴抽出器の初期化
    tda_extractor = TDAFeatureExtractor(
        embedding_dim=3,
        tau=1,
        max_edge_length=2.0,
        persistence_threshold=0.1
    )
    
    # 少数のサンプルで処理（計算時間短縮のため）
    sample_windows = windows[:50]  # 50ウィンドウのみ処理
    sample_labels = window_labels[:50]
    
    # TDA特徴抽出
    try:
        tda_features, components = tda_extractor.extract_features(
            sample_windows, 
            return_components=True
        )
        print(f"TDA特徴形状: {tda_features.shape}")
        print(f"PDスコア形状: {components['pd_scores'].shape}")
        print(f"Simplex tree特徴形状: {components['simplex_tree_features'].shape}")
    
    except Exception as e:
        print(f"TDA特徴抽出エラー: {e}")
        print("ダミーのTDA特徴を生成します...")
        tda_features = np.random.randn(len(sample_windows), 12)
        components = {
            'pd_scores': np.random.randn(len(sample_windows), 1),
            'simplex_tree_features': np.random.randn(len(sample_windows), 11)
        }
    
    # 5. 機械学習モデルの訓練
    print("\n5. 機械学習モデル訓練中...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # 訓練・テストデータ分割
    X_train, X_test, y_train, y_test = train_test_split(
        tda_features, sample_labels, test_size=0.3, random_state=42
    )
    
    # 分類器訓練
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    # 予測
    y_pred = classifier.predict(X_test)
    
    # 6. 評価
    print("\n6. 性能評価中...")
    metrics = evaluate_classification(y_test, y_pred, verbose=True)
    
    # 7. 予測ポイントフィルタリング
    print("\n7. 予測ポイントフィルタリング中...")
    tda_scores = tda_features[:len(y_pred)]  # テストデータに合わせる
    filtered_points = filter_prediction_points(
        X_test, 
        y_pred, 
        tda_scores=tda_scores,
        pd_threshold=0.1,
        density_threshold=5
    )
    print(f"フィルタ済みポイント数: {len(filtered_points)}")
    
    # 8. 分類シグナル生成
    print("\n8. 分類シグナル生成中...")
    signals = create_classification_signals(filtered_points, confidence_threshold=0.5)
    print(f"Class Aシグナル: {len(signals['class_a'])}")
    print(f"Class Bシグナル: {len(signals['class_b'])}")
    print(f"Neutralシグナル: {len(signals['neutral'])}")
    
    # 9. 系列指標（模擬値で）
    print("\n9. 系列指標計算中...")
    mock_values = data[:len(y_pred), 0]  # 主要な値を使用
    sequence_metrics = calculate_sequence_metrics(signals, mock_values)
    print("系列指標:")
    for key, value in sequence_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n=== 実行完了 ===")
    print(f"F1スコア: {metrics['f1_score']:.4f}")
    print("持続ホモロジーを使った分類の基本的な流れを確認できました。")


if __name__ == "__main__":
    main()
