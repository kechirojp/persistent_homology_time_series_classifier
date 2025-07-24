# Persistent Homology Time Series Classifier

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![GUDHI](https://img.shields.io/badge/GUDHI-v3.8.0-green.svg)](https://gudhi.inria.fr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

持続ホモロジーを使った時系列データの分類・異常検知パッケージ

[English README is here / 英語版 README はこちら](README.md)

## 概要

このパッケージは、**持続ホモロジー（Persistent Homology）**を用いた時系列データの分類と異常検知を提供します。TDA（トポロジカルデータ解析）による特徴抽出とPDスコア、シンプレックスツリーの説明変数化を実装しています。

**特に異常検知において優れた性能を発揮**し、従来の統計的手法では検出困難な構造的異常やパターン変化を、データの位相的特徴から効果的に検出できます。

### 🎯 本プロジェクトの特徴

**軽量で実用的な設計**：
- **最小依存関係**: 8パッケージのみで動作（GUDHI + 基本ライブラリ）
- **時系列特化**: 時系列データの異常検知に特化した実装
- **教育的価値**: 持続ホモロジーの基本概念を理解しやすいコード構成

**既存TDAライブラリとの位置づけ**：
- Giotto-TDA/scikit-tda: 研究向け総合ライブラリ（多機能・高機能）
- **本プロジェクト**: 実用向け軽量ライブラリ（シンプル・即座に導入可能）

高度なTDA手法（Persistent Landscape、Betti Curves、Persistent Entropy等）については、[Giotto-TDA](https://giotto-ai.github.io/gtda-docs/)や[scikit-tda](https://scikit-tda.org/)等の専門ライブラリの利用を推奨いたします。

## 主な機能

- **遅延埋め込み**: 時系列データから点群を生成
- **持続ホモロジー計算**: GUDHIを使用したPDスコア抽出
- **シンプレックスツリー特徴**: グラフ理論的統計量の抽出（11次元）
- **データ前処理**: 正負符号付き対数変換、正規化、ウィンドウ化
- **異常検知**: トポロジカル特徴による異常パターン検出
- **分類支援**: 予測ポイントフィルタリング、性能評価

## インストール

### 1. 仮想環境作成（推奨）

```bash
# Python 3.8以上が必要
conda create -n persistent_homology_ts python=3.9 -y
conda activate persistent_homology_ts
```

### 2. パッケージインストール

```bash
pip install -r requirements.txt
```

### 3. 環境チェック

```bash
python check_environment.py
```

## 基本的な使用方法

```python
from src import (
    TDAFeatureExtractor,
    signed_log_transform,
    create_sliding_windows,
    normalize_data
)

# データ前処理
data_transformed = signed_log_transform(data)
data_normalized, _ = normalize_data(data_transformed, method='zscore')
windows, labels = create_sliding_windows(data_normalized, window_size=40)

# TDA特徴抽出
tda_extractor = TDAFeatureExtractor(
    embedding_dim=3,
    tau=1,
    max_edge_length=2.0,
    persistence_threshold=0.1
)

# 特徴抽出実行
tda_features, components = tda_extractor.extract_features(
    windows, 
    return_components=True
)

# PDスコアとSimplex Tree特徴を取得
pd_scores = components['pd_scores']
simplex_features = components['simplex_tree_features']
```

## サンプルデータについて

`example_usage.py`で使用されるサンプルデータは、**お店の来客データ**を模擬したものです：

- **特徴量0-3**: 売上関連指標（メイン売上、関連商品売上など）
- **特徴量4**: ボリューム的特徴（来客数）
  - 好調期・不調期どちらでも「注目度」により来客数が増加
  - `np.abs(pattern)`により絶対値を使用
- **特徴量5**: 指標的特徴（顧客満足度スコア）
  - 好調期は満足度上昇、不調期は満足度下降
  - `np.tanh(pattern)`により-1〜+1の範囲で変動

この設計により、異なる性質の特徴量が組み合わさり、持続ホモロジーによる「形状変化」の検出がより効果的になります。

## 異常検知への応用

```python
# 異常検知用設定
tda_extractor = TDAFeatureExtractor(
    persistence_threshold=0.05  # 異常検知では低めに設定
)

# 異常度判定
pd_scores = components['pd_scores']
anomaly_threshold = np.percentile(pd_scores, 95)  # 上位5%を異常
anomalies = pd_scores > anomaly_threshold

print(f"検出された異常期間: {np.sum(anomalies)} / {len(anomalies)}")
```

## 特徴量詳細

### PDスコア

持続ホモロジーの各次元から計算される重み付き平均スコアです：

**🏝️ 0次元（連結成分・島）- 重み: 0.2（20%）**
- **意味**: データの分離された部分の数と持続時間
- **お店の例**: 営業時間の連続性、基本的な活動状態
- **Simplex Tree関連**: `betti_0`（連結成分数）と直接対応

**🔄 1次元（穴・周期）- 重み: 0.6（60%）← 最重要！**
- **意味**: 周期的パターン、循環構造の安定性
- **お店の例**: 週末/平日サイクル、季節変動、ランチ/ディナー繰り返し
- **Simplex Tree関連**: `betti_1`（1次元穴の数）、`clustering_coeff`と関連

**🌐 2次元（空洞・複雑構造）- 重み: 0.2（20%）**
- **意味**: 高次の複雑な構造パターン
- **お店の例**: 複数周期が重なった複雑なパターン
- **Simplex Tree関連**: `num_triangles`（三角形数）と関連

**計算式**:
```
PD Score = 0.2×(0次元持続) + 0.6×(1次元持続) + 0.2×(2次元持続)
```

**なぜ1次元が60%？**
- 時系列データでは「規則的な繰り返しパターン」が最も重要な情報
- 異常検知では「いつもの周期からの逸脱」を敏感に検出したい
- 0次元・2次元はノイズの影響を受けやすいため重みを抑制

**異常検知での働き**:
- **正常時**: 規則的な周期 → 低いPDスコア
- **異常時**: 周期の崩壊・不規則化 → 高いPDスコア

### Simplex Tree特徴（11次元）

グラフ理論的な統計量で、PDスコアを補完する詳細な構造情報を提供：

**基本構造（PDスコア0次元と関連）:**

1. `num_vertices`: ノード数 - データポイントの総数
2. `num_edges`: エッジ数 - ポイント間の接続数
3. `betti_0`: 連結成分数 - **PDスコア0次元と直接対応**

**周期・穴構造（PDスコア1次元と関連）:**

4. `betti_1`: 1次元穴の数 - **PDスコア1次元と直接対応**
5. `clustering_coeff`: クラスタリング係数 - 局所的な循環構造の密度

**複雑構造（PDスコア2次元と関連）:**

6. `num_triangles`: 三角形数 - 高次構造の基本単位

**グラフ統計量:**

7. `max_degree`: 最大次数 - 最も多く接続されたポイント
8. `avg_degree`: 平均次数 - 平均的な接続密度
9. `avg_filtration`: 平均フィルトレーション - 構造形成の平均閾値
10. `filtration_var`: フィルトレーション分散 - 構造形成の不均一性
11. `max_filtration`: 最大フィルトレーション - 最も遠い接続距離

**PDスコアとの相互補完:**
- PDスコア: 各次元の「持続時間」に重点
- Simplex Tree: 各次元の「個数・密度・分布」に重点
- 両者の組み合わせで、構造の「質」と「量」を同時に評価

## トポロジカル異常検知の利点

- **形状変化の検出**: データの位相的構造変化を捉える
- **ノイズ耐性**: 持続性閾値によるフィルタリング
- **多次元対応**: 高次元時系列データでも効果的

## ファイル構成

```
persistent_homology_time_series_classifier/
├── src/
│   ├── __init__.py                 # パッケージ初期化
│   ├── tda_features.py            # TDA特徴抽出のコア機能
│   ├── data_preprocessing.py      # データ前処理
│   └── classifier_utils.py        # 分類器支援機能
├── tda_extraction_specialized.py  # PDスコア・simplex_tree特化版
├── example_usage.py               # 使用例
├── check_environment.py           # 環境チェック
├── requirements.txt               # 必要ライブラリ
└── README.md                      # このファイル
```

## 実行例

```bash
# 基本的な使用例
python example_usage.py

# 特化版のテスト
python tda_extraction_specialized.py
```

## 必要な依存関係

```
gudhi==3.8.0              # 持続ホモロジー計算
numpy==1.26.4             # 数値計算
scikit-learn==1.4.1.post1 # 機械学習
networkx==3.2.1           # グラフ理論計算
matplotlib==3.8.2         # 可視化
pandas==2.1.4             # データ処理
scipy==1.11.4             # 科学計算
tqdm==4.66.1              # プログレスバー
```

## 適用可能な異常タイプ

- **構造的異常**: データの相関構造や依存関係の変化
- **パターン異常**: 周期性やトレンドの突然の変化  
- **外れ値群**: 個別の外れ値ではなく、異常な期間全体の検出
- **状態変化**: システムの動作状態の変遷点検出

## ライセンス

MIT License

## 貢献

プルリクエストやIssueを歓迎します。

## トラブルシューティング

### よくあるエラー

1. **ImportError: gudhi**: `pip install gudhi` でインストール
2. **メモリ不足**: ウィンドウ数やmax_edge_lengthを調整
3. **計算時間**: step_sizeを大きくして高速化

### サポート

詳細な使用方法については、`example_usage.py`を参照してください。
