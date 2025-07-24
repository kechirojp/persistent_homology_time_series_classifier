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

def generate_sample_data(n_samples: int = 240, n_features: int = 4) -> tuple:
    """
    デパート月次データのサンプル生成（20年分 = 240ヶ月）
    
    【具体例：デパートの月次営業データ】
    - 特徴量0：メイン売上（食品、衣料品、雑貨など）百万円
    - 特徴量1：関連商品売上（イベント商品、季節商品など）百万円  
    - 特徴量2：来客数（月間延べ来客数）千人
    - 特徴量3：営業時間（月間総営業時間）時間
    
    パターン：
    - 通常期：12ヶ月周期の季節変動（春夏秋冬）
    - 特別期：年末年始、GW、お盆の売上増
    - 異常期：コロナ禍（2020年）やリニューアル工事期間
    
    Returns:
    --------
    data : np.ndarray
        デパート月次データ (240, 4)
    labels : np.ndarray  
        期間ラベル (0:通常期, 1:特別期, 2:異常期)
    """
    np.random.seed(42)
    
    # 20年間（240ヶ月）の月インデックス
    months = np.arange(240)
    data = np.zeros((240, 4))
    labels = np.zeros(240, dtype=int)
    
    for i, month in enumerate(months):
        year = 2000 + month // 12  # 2000年から開始
        month_of_year = month % 12 + 1  # 1-12月
        
        # === 基本的な季節パターン ===
        # 季節係数（春夏秋冬）
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * month / 12)
        
        # 年末商戦ブースト（12月、1月）
        if month_of_year in [12, 1]:
            seasonal_factor *= 1.8
            labels[i] = 1  # 特別期
        # GW・お盆ブースト（5月、8月）
        elif month_of_year in [5, 8]:
            seasonal_factor *= 1.4
            labels[i] = 1  # 特別期
        else:
            labels[i] = 0  # 通常期
            
        # === 異常期間の設定 ===
        # コロナ禍（2020年3月-2021年12月）
        if 2020 <= year <= 2021:
            if year == 2020 and month_of_year >= 3:
                seasonal_factor *= 0.4  # 売上大幅減
                labels[i] = 2  # 異常期
            elif year == 2021:
                seasonal_factor *= 0.7  # 回復途上
                labels[i] = 2  # 異常期
                
        # リニューアル工事（2010年6-8月）
        if year == 2010 and month_of_year in [6, 7, 8]:
            seasonal_factor *= 0.2  # 営業停止状態
            labels[i] = 2  # 異常期
            
        # === 各特徴量の生成 ===
        base_noise = np.random.normal(0, 0.1, 4)
        
        # 特徴量0: メイン売上（基準値500百万円）
        data[i, 0] = 500 * seasonal_factor + base_noise[0] * 50
        
        # 特徴量1: 関連商品売上（メイン売上の30-50%）
        relation_ratio = 0.4 + 0.1 * np.sin(2 * np.pi * month / 6)  # 半年周期
        data[i, 1] = data[i, 0] * relation_ratio + base_noise[1] * 20
        
        # 特徴量2: 来客数（基準値100千人）
        customer_factor = seasonal_factor * (1 + 0.2 * np.cos(2 * np.pi * month / 12))
        data[i, 2] = 100 * customer_factor + base_noise[2] * 10
        
        # 特徴量3: 営業時間（基準値300時間/月）
        # 異常期は営業時間短縮
        if labels[i] == 2:  # 異常期
            hour_factor = 0.7  # 営業時間短縮
        else:
            hour_factor = 1.0
        data[i, 3] = 300 * hour_factor + base_noise[3] * 5
    
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
def main():
    """
    デパート月次データを使った持続ホモロジー分析のデモ
    """
    print("🏬 デパート月次データの持続ホモロジー分析デモ")
    print("=" * 60)
    
    # 1. データ生成
    print("1. デパート月次データ生成中...")
    print("   📅 期間: 2000年1月 - 2019年12月（20年間、240ヶ月）")
    print("   📊 特徴量: メイン売上、関連商品売上、来客数、営業時間")
    
    data, labels = generate_sample_data()
    print(f"   データ形状: {data.shape}")
    
    # ラベルの内訳を表示
    label_names = ['通常期', '特別期', '異常期']
    for i, name in enumerate(label_names):
        count = np.sum(labels == i)
        print(f"   {name}: {count}ヶ月")
    
    # 2. データ前処理
    print("\n2. データ前処理中...")
    
    # 対数変換（売上データの歪みを修正）
    data_transformed = signed_log_transform(data)
    print("   ✅ 対数変換完了")
    
    # 正規化
    data_normalized, scaler = normalize_data(data_transformed, method='zscore')
    print("   ✅ Z-score正規化完了")
    
    # 滑らかウィンドウ作成（12ヶ月窓で年単位分析）
    window_size = 12  # 12ヶ月 = 1年間のパターンを見る
    windows, window_labels = create_sliding_windows(
        data_normalized, 
        window_size=window_size,
        step_size=1
    )
    print(f"   ✅ {len(windows)}個の12ヶ月窓を作成")
    
    # 3. TDA特徴抽出
    print("\n3. 持続ホモロジー分析中...")
    
    tda_extractor = TDAFeatureExtractor(
        embedding_dim=3,          # 3次元埋め込み
        tau=1,                    # 遅延時間
        max_edge_length=2.0,      # エッジ最大距離
        persistence_threshold=0.1  # 持続性閾値
    )
    
    print("   🔍 各12ヶ月窓のbirth-death分析実行中...")
    
    try:
        tda_features, components = tda_extractor.extract_features(
            windows[:100],  # 最初の100窓のみ（計算時間短縮）
            return_components=True
        )
        
        print(f"   ✅ TDA特徴抽出完了: {tda_features.shape}")
        
        # 持続性スコアを取得
        persistence_scores = components['pd_scores'].flatten()
        simplex_features = components['simplex_tree_features']
        
        print(f"   📈 持続性スコア: 平均={persistence_scores.mean():.3f}, 標準偏差={persistence_scores.std():.3f}")
        
    except Exception as e:
        print(f"   ❌ TDA特徴抽出エラー: {e}")
        return
    
    # 4. 異常検知分析
    print("\n4. 異常検知分析中...")
    
    # 低い持続性 = 異常と判定
    anomaly_threshold = np.percentile(persistence_scores, 10)  # 下位10%
    anomalies = persistence_scores < anomaly_threshold
    
    print(f"   🎯 異常判定閾値: {anomaly_threshold:.3f}")
    print(f"   🚨 検出された異常窓: {np.sum(anomalies)}/{len(anomalies)}")
    
    # 異常窓の詳細表示
    anomaly_indices = np.where(anomalies)[0]
    if len(anomaly_indices) > 0:
        print("   📋 異常検出された期間:")
        for idx in anomaly_indices[:5]:  # 最初の5つだけ表示
            start_month = idx + 1  # 1始まり
            end_month = start_month + 11
            start_year = 2000 + (start_month - 1) // 12
            end_year = 2000 + (end_month - 1) // 12
            print(f"      - {start_year}年{(start_month-1)%12+1}月〜{end_year}年{(end_month-1)%12+1}月 (持続性: {persistence_scores[idx]:.3f})")
    
    # 5. 構造分析
    print("\n5. データ構造分析...")
    
    # 各次元の統計
    print("   🏝️ 0次元（連結成分）:")
    betti_0 = simplex_features[:, 2]  # betti_0のインデックス
    print(f"      平均: {betti_0.mean():.2f}, 範囲: {betti_0.min():.0f}-{betti_0.max():.0f}")
    
    print("   🔄 1次元（周期構造）:")
    betti_1 = simplex_features[:, 3]  # betti_1のインデックス  
    print(f"      平均: {betti_1.mean():.2f}, 範囲: {betti_1.min():.0f}-{betti_1.max():.0f}")
    
    # 6. 結果サマリー
    print("\n" + "=" * 60)
    print("📊 分析結果サマリー")
    print("=" * 60)
    print(f"🏬 対象期間: 20年間（240ヶ月）のデパート営業データ")
    print(f"🔍 分析窓数: {len(windows)}個の12ヶ月窓")
    print(f"📈 平均持続性: {persistence_scores.mean():.3f}")
    print(f"🚨 異常検出: {np.sum(anomalies)}窓（{100*np.sum(anomalies)/len(anomalies):.1f}%）")
    print(f"⭐ 構造安定性: {'高' if persistence_scores.std() < 0.5 else '中' if persistence_scores.std() < 1.0 else '低'}")
    
    print("\n✅ 分析完了！")
    print("💡 ヒント: 持続性が低い期間は構造的な変化（異常）の可能性があります")
    
    return {
        'data': data,
        'labels': labels,
        'windows': windows,
        'tda_features': tda_features,
        'persistence_scores': persistence_scores,
        'anomalies': anomalies
    }


if __name__ == "__main__":
    results = main()
