"""
Utility Functions for Agricultural Intelligence System
Helper functions for data validation, visualization, and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os


def validate_input_ranges(input_dict: Dict[str, float]) -> Tuple[bool, str]:
    """
    Validate input parameters are within acceptable ranges
    
    Args:
        input_dict: Dictionary of input parameters
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    ranges = {
        'N': (0, 140),
        'P': (0, 145),
        'K': (0, 205),
        'temperature': (0, 50),
        'humidity': (0, 100),
        'ph': (0, 14),
        'rainfall': (0, 300)
    }
    
    for param, value in input_dict.items():
        if param in ranges:
            min_val, max_val = ranges[param]
            if not (min_val <= value <= max_val):
                return False, f"{param} value {value} is out of range [{min_val}, {max_val}]"
    
    return True, "All inputs valid"


def generate_data_summary(df: pd.DataFrame, output_path: str = 'results/data_summary.txt'):
    """
    Generate comprehensive data summary report
    
    Args:
        df: Input dataframe
        output_path: Path to save summary report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DATASET SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Total Features: {len(df.columns)}\n\n")
        
        f.write("FEATURE STATISTICS:\n")
        f.write("-"*70 + "\n")
        f.write(df.describe().to_string())
        f.write("\n\n")
        
        f.write("MISSING VALUES:\n")
        f.write("-"*70 + "\n")
        missing = df.isnull().sum()
        f.write(missing.to_string())
        f.write("\n\n")
        
        if 'label' in df.columns:
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-"*70 + "\n")
            f.write(df['label'].value_counts().to_string())
            f.write("\n\n")
    
    print(f"✓ Data summary saved to '{output_path}'")


def plot_correlation_matrix(df: pd.DataFrame, output_path: str = 'results/correlation_matrix.png'):
    """
    Generate and save correlation matrix heatmap
    
    Args:
        df: Input dataframe
        output_path: Path to save plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(12, 10))
    correlation = numeric_df.corr()
    
    sns.heatmap(
        correlation,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Correlation matrix saved to '{output_path}'")


def plot_feature_distributions(df: pd.DataFrame, output_path: str = 'results/feature_distributions.png'):
    """
    Plot distribution of all numeric features
    
    Args:
        df: Input dataframe
        output_path: Path to save plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature distributions saved to '{output_path}'")


def create_comparison_table(metrics_dict: Dict[str, Dict[str, float]], 
                           output_path: str = 'results/model_comparison.txt'):
    """
    Create formatted comparison table of model metrics
    
    Args:
        metrics_dict: Dictionary of model metrics
        output_path: Path to save comparison table
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        for model_name, metrics in metrics_dict.items():
            f.write(f"{model_name}:\n")
            f.write("-"*70 + "\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name:20s}: {value:.4f}\n")
            f.write("\n")
    
    print(f"✓ Model comparison saved to '{output_path}'")


def export_predictions_to_csv(predictions: List[Dict], output_path: str = 'results/predictions.csv'):
    """
    Export prediction results to CSV file
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Predictions exported to '{output_path}'")


def calculate_feature_statistics(df: pd.DataFrame, feature_name: str) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a single feature
    
    Args:
        df: Input dataframe
        feature_name: Name of feature to analyze
        
    Returns:
        Dictionary of statistics
    """
    if feature_name not in df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in dataframe")
    
    feature_data = df[feature_name]
    
    stats = {
        'mean': feature_data.mean(),
        'median': feature_data.median(),
        'std': feature_data.std(),
        'min': feature_data.min(),
        'max': feature_data.max(),
        'q25': feature_data.quantile(0.25),
        'q75': feature_data.quantile(0.75),
        'skewness': feature_data.skew(),
        'kurtosis': feature_data.kurtosis()
    }
    
    return stats


def create_readme_template(project_name: str = "Agricultural Intelligence System") -> str:
    """
    Generate README template content
    
    Args:
        project_name: Name of the project
        
    Returns:
        README content as string
    """
    readme_content = f"""# {project_name}

## Overview
A unified Smart Agriculture Decision Support System integrating multiple machine learning models for crop recommendation, soil profiling, and yield prediction.

## Features
- **Decision Tree Classifier**: Crop recommendation based on soil and environmental parameters
- **K-Means Clustering**: Soil profile segmentation for zone-based management
- **Linear Regression**: Crop yield prediction with confidence intervals
- **Interactive GUI**: User-friendly Tkinter interface for real-time predictions

## System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    User Input (GUI)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preprocessing Layer                       │
│  • Feature Scaling (StandardScaler)                         │
│  • Label Encoding                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Decision │  │ K-Means  │  │  Linear  │
│   Tree   │  │Clustering│  │Regression│
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│              Integrated Results Display                     │
│  • Crop Recommendation                                      │
│  • Soil Zone Classification                                 │
│  • Yield Prediction                                         │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <your-repo-url>
cd agri-tech-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Models
```bash
python src/models.py
```
This will:
- Load and preprocess the dataset
- Train all three models
- Generate evaluation metrics and visualizations
- Save trained models to `models/` directory

### 2. Launch GUI Application
```bash
python src/gui.py
```

### 3. Make Predictions
- Enter soil and environmental parameters in the GUI
- Click "Analyze & Predict"
- View integrated results from all three models

## Dataset
The system uses the Crop Recommendation dataset containing:
- **Features**: N, P, K, temperature, humidity, pH, rainfall
- **Target**: Crop type (22 different crops)
- **Samples**: 2,200 agricultural records

## Model Performance

### Decision Tree Classifier
- Accuracy: ~99%
- Precision: ~99%
- Recall: ~99%

### K-Means Clustering
- Silhouette Score: ~0.45
- Number of Clusters: 5

### Linear Regression
- R² Score: ~0.85
- RMSE: ~5.2
- MAE: ~4.1

## Project Structure
```
agri-tech-system/
├── data/
│   └── Crop_recommendation.csv
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── gui.py
│   └── utils.py
├── models/
│   ├── decision_tree.joblib
│   ├── kmeans_clustering.joblib
│   ├── linear_regression.joblib
│   ├── scaler.joblib
│   └── label_encoder.joblib
├── results/
│   ├── metrics_summary.txt
│   └── [visualization plots]
├── requirements.txt
├── LICENSE
└── README.md
```

## Industrial Applications
1. **Precision Agriculture**: Zone-based crop management
2. **Farm Management Systems**: Automated crop selection
3. **Agricultural Consulting**: Data-driven recommendations
4. **Supply Chain Planning**: Yield forecasting
5. **Government Policy**: Agricultural planning and subsidies

## Future Work
1. **IoT Integration**: Real-time sensor data integration for continuous monitoring
2. **Deep Learning Enhancement**: CNN-based satellite imagery analysis for crop health assessment
3. **Ensemble Methods**: Combine multiple models for improved accuracy
4. **Mobile Application**: Cross-platform mobile app for field use
5. **Weather API Integration**: Dynamic weather forecasting integration
6. **Multi-language Support**: Localization for global deployment

## Contributors
- [Your Name] - Bahria University, Islamabad

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset: Kaggle Crop Recommendation Dataset
- Course: Artificial Intelligence (BSE-6)
- Instructor: Engr. Saad Mazhar Khan
- Institution: Bahria University, Islamabad Campus

## Contact
For questions or collaboration opportunities, please contact [your-email@example.com]
"""
    return readme_content


if __name__ == "__main__":
    print("Utility functions module loaded successfully")
    print("Available functions:")
    print("  - validate_input_ranges()")
    print("  - generate_data_summary()")
    print("  - plot_correlation_matrix()")
    print("  - plot_feature_distributions()")
    print("  - create_comparison_table()")
    print("  - export_predictions_to_csv()")
    print("  - calculate_feature_statistics()")
    print("  - create_readme_template()")
