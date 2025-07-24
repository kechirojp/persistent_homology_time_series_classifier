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

# Data preprocessing for monthly department store data
data_transformed = signed_log_transform(data)
data_normalized, _ = normalize_data(data_transformed, method='zscore')
windows, labels = create_sliding_windows(data_normalized, window_size=12)  # 12-month windows

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

# Get birth-death distances and Simplex Tree features
persistence_scores = components['pd_scores']  # Persistence duration for each structure
simplex_features = components['simplex_tree_features']
```

## Sample Data

The sample data used in `example_usage.py` simulates **monthly department store business data**:

- **Feature 0**: Main sales (food, clothing, miscellaneous goods, etc.) in millions of yen
- **Feature 1**: Related product sales (event products, seasonal products, etc.) in millions of yen
- **Feature 2**: Customer count (monthly total visitors) in thousands
- **Feature 3**: Operating hours (monthly total operating hours) in hours

**Data Characteristics**:
- 12-month seasonal patterns (spring, summer, autumn, winter sales variations)
- Variations during special periods (New Year, Golden Week, Obon, etc.)
- Mix of normal and abnormal periods (COVID-19, major construction, etc.)

Using such **familiar and understandable data** makes it intuitive to understand "structural changes" in persistent homology.

## Anomaly Detection Application

**Basic Concept**:
- **Normal months**: Same seasonal patterns every year → Structures last long → High persistence
- **Abnormal months**: Patterns break down → Structures collapse quickly → Low persistence

```python
# Detect months with low persistence as anomalies
tda_extractor = TDAFeatureExtractor()
features, components = tda_extractor.extract_features(windows, return_components=True)

# Persistence scores (high = normal, low = abnormal)
persistence_scores = components['pd_scores']

# Classify bottom 10% as anomalies
anomaly_threshold = np.percentile(persistence_scores, 10)
anomalies = persistence_scores < anomaly_threshold

print(f"Abnormal periods: {np.sum(anomalies)} / {len(anomalies)} windows")

# Identify which months are abnormal
for i, is_anomaly in enumerate(anomalies):
    if is_anomaly:
        print(f"Anomaly detected in {i+1}th 12-month window")
```

**Department Store Example**:
- Normal: December every year shows regular year-end sales increases
- Abnormal: December 2020 during COVID-19 shows completely different patterns → Detected by low persistence

## What is Persistent Homology?

**Simply put, it's a technique to measure "when structures begin and end"**

### Basic Concept: Birth-Death

Track "when certain structures start and when they end" in data:

**📍 Birth**: The moment when a new structure appears
- Department store example: "Period when sales start rising rapidly", "Start of new customer pattern"

**⚰️ Death**: The moment when that structure disappears  
- Department store example: "Period when sales rise ends", "End of customer pattern"

**⏱️ Persistence**: Death - Birth
- "How long that structure lasted"
- Long-lasting structures = Important patterns
- Quickly disappearing structures = Noise

### Viewing Structures in 3 Dimensions

**🏝️ 0-dimension (Islands/Connected Components)**
- How many "clusters" of data exist
- Department store example: "Separation of weekday and weekend customers", "Different generational customer groups"

**🔄 1-dimension (Holes/Cycles)**  
- "Repetitive patterns" in data
- Department store example: "Month-end sale cycles", "Seasonal variation patterns"

**🌐 2-dimension (Voids/Complex Structures)**
- Complex 3-dimensional structures
- Department store example: "Complex sales patterns involving multiple factors"

### Why Effective for Anomaly Detection?

**Normal times**: Regular patterns → Structures persist long (long persistence)
**Abnormal times**: Patterns collapse → Structures disappear quickly (short persistence)

**Concrete Example (Department Store)**:
- Normal: Sales rise at month-end every month (12 regular birth-death cycles)
- Abnormal: Pattern disappears during COVID-19 (disrupted birth-death)

## Feature Details

### Birth-Death Distance (Persistence Distance)

Measures the "length of persistence duration" for each structure:

```python
persistence_distance = death_value - birth_value
```

**Long persistence duration** = Stable important structure (normal pattern)
**Short persistence duration** = Unstable noise (sign of anomaly)

### Simplex Tree Features (11 dimensions)

Auxiliary indicators that reveal "structural details" not visible from Birth-Death alone:

**📊 Basic Structural Counts**
1. `num_vertices`: Total number of data points
2. `num_edges`: Number of connections between points  
3. `betti_0`: Number of separated groups

**🔄 Repetitive Pattern Details**
4. `betti_1`: Number of cyclic structures
5. `clustering_coeff`: Density of local connections

**🌐 Complex Structures**
6. `num_triangles`: Number of triangular structures

**📈 Network Statistics**
7. `max_degree`: Most connected point
8. `avg_degree`: Average number of connections
9. `avg_filtration`: Average timing of structure formation
10. `filtration_var`: Variation in structure formation
11. `max_filtration`: Most distant connection

**Department Store Interpretation Examples**:
- High `betti_1` = Regular seasonal patterns exist
- High `clustering_coeff` = Customer groups are densely related
- Abnormally high `max_degree` = Everything concentrated in specific month (sign of anomaly)

### Practical Analysis Steps

**Step 1: Data Preparation**
```python
# Monthly department store data (24 months)
data = np.array([
    [sales1, sales2, customers, hours],  # January
    [sales1, sales2, customers, hours],  # February
    # ... 24 months of data
])
```

**Step 2: Preprocessing**
```python
# Smooth and normalize data
data_transformed = signed_log_transform(data)
data_normalized, _ = normalize_data(data_transformed)
windows, labels = create_sliding_windows(data_normalized, window_size=12)  # 12-month windows
```

**Step 3: Persistent Homology Analysis**
```python
# Birth-death analysis for each window
tda_extractor = TDAFeatureExtractor()
features, components = tda_extractor.extract_features(windows, return_components=True)

# Check "structural stability" for each month
persistence_scores = components['pd_scores']
simplex_features = components['simplex_tree_features']
```

**Step 4: Anomaly Month Detection**
```python
# Abnormally short persistence = Unstable structure = Anomaly month
threshold = np.percentile(persistence_scores, 5)  # Bottom 5%
anomaly_months = persistence_scores < threshold

print(f"Anomaly months: {np.where(anomaly_months)[0] + 1}th month")
```

This approach allows **intuitive understanding of "why that month is abnormal" from a structural perspective**.

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
