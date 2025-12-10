# transfer-compat

A lightweight Python library for dataset compatibility analysis, preprocessing alignment, drift detection, and visualization — designed to help compare two datasets (source vs target) before transfer learning or model migration.

## Features

### 🔍 Core Comparison
- Detect shared features  
- Compare schema and type alignment  
- Compute a compatibility score  

### 🛠 Preprocessing
- Automatic type harmonization  
- Missing-value alignment  
- Normalization across datasets  
- Outlier smoothing  

### 📊 Visualization
- Distribution comparison (hist/KDE + categorical bars)
- Wasserstein distance trend
- Drift radar chart
- Domain-overlap heatmap

### 📈 Metrics
- Wasserstein distance  
- Categorical drift  
- Overlap measures  

---

## Installation

Clone the repo and install locally:

```bash
git clone https://github.com/najmehjabbari2024-deln/transfer-compat
cd transfer-compat
pip install -e .
 
