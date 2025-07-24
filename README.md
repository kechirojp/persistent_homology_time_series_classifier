# Persistent Homology Time Series Classifier

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![GUDHI](https://img.shields.io/badge/GUDHI-v3.8.0-green.svg)](https://gudhi.inria.fr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A time series classification and anomaly detection package using Persistent Homology.

[日本語版 README はこちら / Japanese README is here](README_ja.md)

## Overview

This package provides time series data classification and anomaly detection using **Persistent Homology**. It implements feature extraction through TDA (Topological Data Analysis) with PD scores and simplex tree explanatory variables.

**Particularly excellent performance in anomaly detection**, effectively detecting structural anomalies and pattern changes that are difficult to detect with conventional statistical methods through topological features of data.

### 🎯 Project Characteristics

**Lightweight and Practical Design**:
- **Minimal Dependencies**: Operates with 8 packages only (GUDHI + basic libraries)
- **Time Series Specialized**: Implementation specialized for time series anomaly detection
- **Educational Value**: Code structure designed for easy understanding of basic persistent homology concepts

**Positioning with Existing TDA Libraries**:
- Giotto-TDA/scikit-tda: Research-oriented comprehensive libraries (feature-rich and sophisticated)
- **This Project**: Practice-oriented lightweight library (simple and readily deployable)

For advanced TDA methods (Persistent Landscape, Betti Curves, Persistent Entropy, etc.), we recommend utilizing specialized libraries such as [Giotto-TDA](https://giotto-ai.github.io/gtda-docs/) or [scikit-tda](https://scikit-tda.org/).

## Key Features

- **Delay Embedding**: Generate point clouds from time series data
- **Persistent Homology Computation**: PD score extraction using GUDHI
- **Simplex Tree Features**: Graph-theoretic statistical quantity extraction (11 dimensions)
- **Data Preprocessing**: Signed log transformation, normalization, windowing
- **Anomaly Detection**: Anomaly pattern detection through topological features
- **Classification Support**: Prediction point filtering, performance evaluation

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Python 3.8 or higher required
conda create -n persistent_homology_ts python=3.9 -y
conda activate persistent_homology_ts
```

### 2. Install Packages

```bash
pip install -r requirements.txt
```

### 3. Environment Check

```bash
python check_environment.py
```

## Basic Usage

```python
from src import (
    TDAFeatureExtractor,
    signed_log_transform,
    create_sliding_windows,
    normalize_data
)

# Data preprocessing
data_transformed = signed_log_transform(data)
data_normalized, _ = normalize_data(data_transformed, method='zscore')
windows, labels = create_sliding_windows(data_normalized, window_size=40)

# TDA feature extraction
tda_extractor = TDAFeatureExtractor(
    embedding_dim=3,
    tau=1,
    max_edge_length=2.0,
    persistence_threshold=0.1
)

# Execute feature extraction
tda_features, components = tda_extractor.extract_features(
    windows, 
    return_components=True
)

# Get PD scores and Simplex Tree features
pd_scores = components['pd_scores']
simplex_features = components['simplex_tree_features']
```

## Sample Data

The sample data used in `example_usage.py` simulates **store visitor data**:

- **Features 0-3**: Sales-related indicators (main sales, related product sales, etc.)
- **Feature 4**: Volume-based feature (visitor count)
  - Increases during both good and bad periods due to "attention level"
  - Uses `np.abs(pattern)` for absolute values
- **Feature 5**: Indicator-based feature (customer satisfaction score)
  - Rises during good periods, falls during bad periods
  - Uses `np.tanh(pattern)` to vary within -1 to +1 range

This design combines features with different characteristics, making topological "shape change" detection through persistent homology more effective.

## Anomaly Detection Application

```python
# Anomaly detection configuration
tda_extractor = TDAFeatureExtractor(
    persistence_threshold=0.05  # Set lower for anomaly detection
)

# Anomaly assessment
pd_scores = components['pd_scores']
anomaly_threshold = np.percentile(pd_scores, 95)  # Top 5% as anomalies
anomalies = pd_scores > anomaly_threshold

print(f"Detected anomaly periods: {np.sum(anomalies)} / {len(anomalies)}")
```

## Feature Details

### PD Score

Weighted average score calculated from each dimension of persistent homology:

**🏝️ 0-dimension (Connected Components/Islands) - Weight: 0.2 (20%)**
- **Meaning**: Number and persistence of separated data parts
- **Store Example**: Business hours continuity, basic activity state
- **Simplex Tree Relation**: Directly corresponds to `betti_0` (connected components)

**🔄 1-dimension (Holes/Cycles) - Weight: 0.6 (60%) ← Most Important!**
- **Meaning**: Periodic patterns, stability of cyclic structures
- **Store Example**: Weekend/weekday cycles, seasonal variations, lunch/dinner repetitions
- **Simplex Tree Relation**: Related to `betti_1` (1-dimensional holes), `clustering_coeff`

**🌐 2-dimension (Voids/Complex Structures) - Weight: 0.2 (20%)**
- **Meaning**: Higher-order complex structural patterns
- **Store Example**: Complex patterns with multiple overlapping cycles
- **Simplex Tree Relation**: Related to `num_triangles` (triangle count)

**Formula**:
```
PD Score = 0.2×(0-dim persistence) + 0.6×(1-dim persistence) + 0.2×(2-dim persistence)
```

**Why 60% for 1-dimension?**
- "Regular repetitive patterns" are most important information in time series data
- For anomaly detection, we want to sensitively detect "deviations from usual cycles"
- 0-dim and 2-dim are more susceptible to noise, so weights are suppressed

**Role in Anomaly Detection**:
- **Normal**: Regular cycles → Low PD score
- **Anomalous**: Cycle collapse/irregularity → High PD score

### Simplex Tree Features (11 dimensions)

Graph-theoretic statistics providing detailed structural information to complement PD scores:

**Basic Structure (Related to PD Score 0-dimension):**

1. `num_vertices`: Number of nodes - Total count of data points
2. `num_edges`: Number of edges - Count of connections between points
3. `betti_0`: Number of connected components - **Directly corresponds to PD Score 0-dimension**

**Cycle/Hole Structure (Related to PD Score 1-dimension):**

4. `betti_1`: Number of 1-dimensional holes - **Directly corresponds to PD Score 1-dimension**
5. `clustering_coeff`: Clustering coefficient - Density of local cyclic structures

**Complex Structure (Related to PD Score 2-dimension):**

6. `num_triangles`: Number of triangles - Basic units of higher-order structures

**Graph Statistics:**

7. `max_degree`: Maximum degree - Most connected point
8. `avg_degree`: Average degree - Average connection density
9. `avg_filtration`: Average filtration - Average threshold for structure formation
10. `filtration_var`: Filtration variance - Irregularity in structure formation
11. `max_filtration`: Maximum filtration - Most distant connection distance

**Complementary Relationship with PD Score:**
- PD Score: Focuses on "persistence duration" of each dimension
- Simplex Tree: Focuses on "count, density, distribution" of each dimension
- Combined evaluation of structural "quality" and "quantity"

## Advantages of Topological Anomaly Detection

- **Shape Change Detection**: Captures topological structural changes in data
- **Noise Resistance**: Filtering through persistence threshold
- **Multi-dimensional Support**: Effective even with high-dimensional time series data

## File Structure

```
persistent_homology_time_series_classifier/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── tda_features.py            # TDA feature extraction core functionality
│   ├── data_preprocessing.py      # Data preprocessing
│   └── classifier_utils.py        # Classifier support functions
├── tda_extraction_specialized.py  # PD score & simplex_tree specialized version
├── example_usage.py               # Usage example
├── check_environment.py           # Environment check
├── requirements.txt               # Required libraries
├── README.md                      # This file (English)
└── README_ja.md                   # Japanese README
```

## Examples

```bash
# Basic usage example
python example_usage.py

# Specialized version test
python tda_extraction_specialized.py
```

## Required Dependencies

```
gudhi==3.8.0              # Persistent homology computation
numpy==1.26.4             # Numerical computation
scikit-learn==1.4.1.post1 # Machine learning
networkx==3.2.1           # Graph theory computation
matplotlib==3.8.2         # Visualization
pandas==2.1.4             # Data processing
scipy==1.11.4             # Scientific computation
tqdm==4.66.1              # Progress bar
```

## Applicable Anomaly Types

- **Structural Anomalies**: Changes in data correlation structure and dependencies
- **Pattern Anomalies**: Sudden changes in periodicity and trends
- **Outlier Groups**: Detection of entire anomalous periods rather than individual outliers
- **State Changes**: Detection of system operational state transition points

## License

MIT License

## Contributing

Pull requests and issues are welcome.

## Troubleshooting

### Common Errors

1. **ImportError: gudhi**: Install with `pip install gudhi`
2. **Memory shortage**: Adjust number of windows or max_edge_length
3. **Computation time**: Increase step_size for acceleration

### Support

For detailed usage, please refer to `example_usage.py`.
