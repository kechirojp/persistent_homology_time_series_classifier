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

# データ前処理（デパート月次データ用）
data_transformed = signed_log_transform(data)
data_normalized, _ = normalize_data(data_transformed, method='zscore')
windows, labels = create_sliding_windows(data_normalized, window_size=12)  # 12ヶ月窓

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

# Birth-Death距離とSimplex Tree特徴を取得
persistence_scores = components['pd_scores']  # 各構造の持続時間
simplex_features = components['simplex_tree_features']
```

## サンプルデータについて

`example_usage.py`で使用されるサンプルデータは、**月ごとのデパートの営業データ**を模擬したものです：

- **特徴量0**: メイン売上（食品、衣料品、雑貨など）
- **特徴量1**: 関連商品売上（イベント商品、季節商品など）
- **特徴量2**: 来客数（月ごとの延べ来客数）
- **特徴量3**: 営業時間（月ごとの総営業時間）

**データの特徴**：
- 12ヶ月周期の季節パターン（春夏秋冬の売上変動）
- 年末年始・GW・お盆などの特別期間での変動
- 通常期と異常期（コロナ禍、大規模工事など）の混在

このような**身近で理解しやすいデータ**を使うことで、持続ホモロジーの「構造の変化」を直感的に理解できます。

## 異常検知への応用

**基本的な考え方**：
- **正常な月**: 毎年同じような季節パターン → 構造が長く続く → 高い持続性
- **異常な月**: パターンが崩れる → 構造がすぐ壊れる → 低い持続性

```python
# 持続性の低い月を異常として検出
tda_extractor = TDAFeatureExtractor()
features, components = tda_extractor.extract_features(windows, return_components=True)

# 持続性スコア（高い = 正常、低い = 異常）
persistence_scores = components['pd_scores']

# 下位10%を異常と判定
anomaly_threshold = np.percentile(persistence_scores, 10)
anomalies = persistence_scores < anomaly_threshold

print(f"異常な期間: {np.sum(anomalies)} / {len(anomalies)}窓")

# どの月が異常かを特定
for i, is_anomaly in enumerate(anomalies):
    if is_anomaly:
        print(f"{i+1}番目の12ヶ月窓で異常を検出")
```

**デパートでの実例**：
- 正常：毎年12月は年末商戦で規則的な売上増加
- 異常：コロナ禍の2020年12月は例年と全く違うパターン → 低い持続性で検出

## 持続ホモロジーとは？

**簡単に言うと「構造の始まりと終わり」を測る技術です**

### 基本概念：Birth-Death（誕生-消滅）

データの中で「ある構造がいつ始まって、いつ終わるか」を追跡します：

**📍 Birth（誕生）**: 新しい構造が現れる瞬間
- デパートの例：「売上が急上昇し始める時期」「新しい客層パターンの開始」

**⚰️ Death（消滅）**: その構造が消える瞬間  
- デパートの例：「売上上昇が終わる時期」「客層パターンの終了」

**⏱️ Persistence（持続時間）**: Death - Birth
- 「その構造がどのくらい長続きしたか」
- 長く続く構造 = 重要なパターン
- すぐ消える構造 = ノイズ

### 3つの次元で構造を見る

**🏝️ 0次元（島・連結成分）**
- データの「塊」がいくつあるか
- デパート例：「平日客と休日客の分離」「異なる世代の客層」

**🔄 1次元（穴・周期）**  
- データの「繰り返しパターン」
- デパート例：「月末セールの周期」「季節変動パターン」

**🌐 2次元（空洞・複雑構造）**
- 複雑な3次元的構造
- デパート例：「複数要因が絡んだ複雑な売上パターン」

### なぜ異常検知に効果的？

**正常時**: 規則的なパターン → 構造が長く持続（長いpersistence）
**異常時**: パターンが崩れる → 構造がすぐ消滅（短いpersistence）

**具体例（デパート）**：
- 正常：毎月末セールで売上上昇（12回の規則的birth-death）
- 異常：コロナ禍でパターン消滅（birth-deathの乱れ）

## 特徴量詳細

### Birth-Death Distance（持続距離）

各構造の「持続時間の長さ」を測定：

```python
persistence_distance = death_value - birth_value
```

**長い持続時間** = 安定した重要な構造（正常パターン）
**短い持続時間** = 不安定なノイズ（異常の兆候）

### Simplex Tree特徴（11次元）

Birth-Deathだけでは見えない「構造の詳細」を教えてくれる補助指標：

**📊 基本的な構造の数**
1. `num_vertices`: データポイントの総数
2. `num_edges`: ポイント同士の繋がりの数  
3. `betti_0`: 分離されたグループの数

**🔄 繰り返しパターンの詳細**
4. `betti_1`: 周期構造の数
5. `clustering_coeff`: 局所的な繋がりの密度

**🌐 複雑な構造**
6. `num_triangles`: 三角形構造の数

**📈 ネットワーク統計**
7. `max_degree`: 最も多く繋がったポイント
8. `avg_degree`: 平均的な繋がり数
9. `avg_filtration`: 構造形成の平均タイミング
10. `filtration_var`: 構造形成のバラつき
11. `max_filtration`: 最も遠い繋がり

**デパートでの解釈例**：
- `betti_1`が高い = 規則的な季節パターンがある
- `clustering_coeff`が高い = 客層が密に関連している
- `max_degree`が異常に高い = 特定の月に全てが集中している（異常の兆候）

### 実際の分析手順

**ステップ1: データの準備**
```python
# 月ごとのデパートデータ（24ヶ月分）
data = np.array([
    [売上1, 売上2, 来客数, 営業時間],  # 1月
    [売上1, 売上2, 来客数, 営業時間],  # 2月
    # ... 24ヶ月分
])
```

**ステップ2: 前処理**
```python
# データを滑らかにして、正規化
data_transformed = signed_log_transform(data)
data_normalized, _ = normalize_data(data_transformed)
windows, labels = create_sliding_windows(data_normalized, window_size=12)  # 12ヶ月窓
```

**ステップ3: 持続ホモロジー分析**
```python
# 各窓でbirth-death分析
tda_extractor = TDAFeatureExtractor()
features, components = tda_extractor.extract_features(windows, return_components=True)

# 各月の「構造の安定性」を確認
persistence_scores = components['pd_scores']
simplex_features = components['simplex_tree_features']
```

**ステップ4: 異常月の検出**
```python
# 異常に短い持続時間 = 構造が不安定 = 異常月
threshold = np.percentile(persistence_scores, 5)  # 下位5%
anomaly_months = persistence_scores < threshold

print(f"異常な月: {np.where(anomaly_months)[0] + 1}月")
```

この方法なら、**「なぜその月が異常なのか」が構造の視点から直感的に理解**できます。

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
├── README.md                      # 英語版README
└── README_ja.md                   # 日本語版README（このファイル）
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
