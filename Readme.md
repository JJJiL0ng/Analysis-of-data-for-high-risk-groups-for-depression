# Mental Health Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-red)](https://xgboost.readthedocs.io/)

## 🎯 Project Overview

This project investigates the relationship between lifestyle factors and mental health conditions, specifically focusing on anxiety and depression prediction using machine learning models. Our goal is to build predictive models that can identify individuals at high risk for mental health issues, enabling early intervention and support.

## 🔍 Business Objective

- **Primary Goal**: Identify high-risk groups for depression by analyzing various lifestyle and demographic features
- **Clinical Impact**: Enable early intervention through predictive modeling before symptoms become severe
- **Support System**: Assist mental health professionals in risk assessment and resource allocation
- **Prevention Focus**: Recommend counseling or treatment for at-risk individuals before it's too late

## 📊 Dataset Information

- **Source**: [Kaggle - Anxiety and Depression Mental Health Factors](https://www.kaggle.com/datasets/ak0212/anxiety-and-depression-mental-health-factors)
- **File**: `anxiety_depression_data.csv`
- **Size**: 1,200 samples × 21 features
- **Target Variable**: Depression levels (Low: ≤9, Medium: 10-14, High: ≥15)

### Feature Categories
- **Demographics**: Age, Gender, Education Level, Employment Status
- **Lifestyle**: Sleep Hours, Physical Activity, Substance Use, Medication Use
- **Mental Health Indicators**: Anxiety Score, Depression Score, Stress Level
- **Psychosocial Factors**: Social Support, Self-Esteem, Life Satisfaction, Loneliness
- **Health Conditions**: Family History, Chronic Illnesses, Therapy Status

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pyenv (recommended for environment management)
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mental-health-prediction.git
   cd mental-health-prediction
   ```

2. **Set up Python environment with pyenv** (recommended):
   ```bash
   # Install Python 3.9 (if not already installed)
   pyenv install 3.9.18
   
   # Set local Python version for this project
   pyenv local 3.9.18
   
   # Create virtual environment
   python -m venv mental_health_env
   
   # Activate virtual environment
   # On macOS/Linux:
   source mental_health_env/bin/activate
   # On Windows:
   mental_health_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Upgrade pip first
   pip install --upgrade pip
   
   # Install required packages
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn plotly scipy
   ```

4. **Download the dataset**:
   ```bash
   # Download anxiety_depression_data.csv from Kaggle and place in root directory
   # Link: https://www.kaggle.com/datasets/ak0212/anxiety-and-depression-mental-health-factors
   ```

### Usage

**🎯 One-Command Execution (Recommended)**:
```bash
# Run the complete pipeline with one command
python main_code.py
```

This will automatically execute:
1. Data encoding (`encoding.py`)
2. Data scaling (`scaling.py`) 
3. Classification models and evaluation (`evaluation_kfold.py`)
4. K-means clustering analysis (`K-means.py`)

**Manual Execution** (if needed):
```bash
# Run individual modules
python encoding.py
python scaling.py
python evaluation_kfold.py
python K-means.py
```

### Expected Output
After successful execution, you'll find:
```
results/
├── encoded_data.csv                    # Processed categorical data
├── Robustscaling_Q1.csv               # Scaled numerical data
├── kmeans_clustering_results.csv       # Clustering analysis results
└── pipeline_execution_summary.txt      # Complete execution summary
```

Plus various visualization plots displayed during execution.

## 🏗️ Architecture

The project follows a modular architecture designed for scalability and maintainability:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  Jupyter Notebooks  │  Visualization  │  Reports & Results  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  Classification Models  │  Clustering Analysis │ Evaluation │
│  • Decision Tree        │  • K-means Clustering│ • Cross-   │
│  • Random Forest        │  • Silhouette Score  │   validation│
│  • XGBoost              │  • Elbow Method      │ • Metrics  │
│  • Bagging Classifier   │                      │ • Reports  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  Data Preprocessing  │  Feature Engineering │ Model Training│
│  • Data Cleaning     │  • Feature Selection │ • Hyperparameter│
│  • Encoding          │  • Scaling           │   Tuning      │
│  • Outlier Handling  │  • SMOTE/SMOTENC     │ • Threshold   │
│                      │                      │   Optimization│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
├─────────────────────────────────────────────────────────────┤
│    Raw Data Storage   │    Processed Data    │  Model Storage│
│  • anxiety_depression │  • Encoded Features  │ • Trained    │
│    _data.csv          │  • Scaled Data       │   Models     │
│  • Original Features  │  • Target Variables  │ • Configs    │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Data Processing Pipeline
- **Input**: Raw CSV data with mixed feature types
- **Cleaning**: Handle missing values, outliers, and inconsistent categories
- **Encoding**: Ordinal encoding for ordered categories, One-hot encoding for nominal features
- **Scaling**: Robust scaling to handle outliers effectively
- **Output**: Clean, encoded, and scaled feature matrix

#### 2. Machine Learning Pipeline
- **Classification**: Binary and multi-class depression risk prediction
- **Clustering**: Unsupervised analysis to identify natural groupings
- **Evaluation**: Comprehensive model assessment with cross-validation
- **Optimization**: Hyperparameter tuning and threshold optimization

#### 3. Analysis Framework
- **Feature Importance**: Identify key predictors of mental health risk
- **Model Comparison**: Evaluate multiple algorithms systematically
- **Clinical Interpretation**: Translate results into actionable insights

## 🛠️ Key Features

### Data Preprocessing
- **Robust Data Cleaning**: Handle missing values and inconsistent entries
- **Advanced Encoding**: Ordinal and one-hot encoding for categorical variables
- **Outlier-Resistant Scaling**: Robust scaling using interquartile range
- **Class Balancing**: SMOTE and SMOTENC for imbalanced datasets

### Machine Learning Models
- **Decision Tree Classifier**: Interpretable model with feature importance
- **Random Forest**: Ensemble method for improved accuracy
- **XGBoost**: Gradient boosting for high performance
- **Bagging Classifier**: Bootstrap aggregating for variance reduction

### Advanced Analytics
- **K-means Clustering**: Unsupervised pattern discovery
- **Feature Selection**: Importance-based dimensionality reduction
- **Threshold Tuning**: Optimize classification thresholds for clinical needs
- **Cross-Validation**: 5-fold validation for robust performance estimation

### Visualization & Reporting
- **Feature Importance Plots**: Identify key predictors
- **Clustering Visualizations**: PCA-based cluster analysis
- **Performance Metrics**: Comprehensive model evaluation
- **Clinical Reports**: Interpretable results for healthcare professionals

## 🏆 Top 5 Best Model Combinations

Based on comprehensive analysis using our modular framework that tests multiple preprocessing methods, model algorithms, and evaluation metrics, here are the **top 5 performing combinations** for mental health risk prediction:

### 🥇 **Rank 1: Random Forest + Robust Scaling**
- **Preprocessing**: Robust Scaling + Ordinal & One-hot Encoding
- **Model**: Random Forest Classifier
- **Parameters**: 
  - n_estimators: 50
  - max_depth: 8
  - max_features: sqrt
  - criterion: gini
- **Evaluation**: Holdout validation
- **Performance**: 
  - **Accuracy: 0.428**
- **Strengths**: Best overall performance with robust scaling, handles outliers effectively

### 🥈 **Rank 2: Random Forest + Z-Score Scaling**
- **Preprocessing**: Z-Score (Standard) Scaling + Encoding
- **Model**: Random Forest Classifier
- **Parameters**: 
  - n_estimators: 100
  - max_depth: 8
  - max_features: log2
  - criterion: gini
- **Evaluation**: Stratified K-fold Cross-validation
- **Performance**:
  - **Accuracy: 0.427**
- **Strengths**: Excellent cross-validation performance, standard preprocessing approach

### 🥉 **Rank 3: Decision Tree + Robust Scaling**
- **Preprocessing**: Robust Scaling + Encoding
- **Model**: Decision Tree Classifier
- **Parameters**: 
  - max_depth: 3
  - criterion: gini
  - splitter: best
- **Evaluation**: Holdout validation
- **Performance**:
  - **Accuracy: 0.424**
- **Strengths**: High interpretability, simple tree structure, robust to outliers

### 🎖️ **Rank 4: Decision Tree + Z-Score Scaling (Entropy)**
- **Preprocessing**: Z-Score Scaling + Encoding
- **Model**: Decision Tree Classifier
- **Parameters**: 
  - max_depth: 3
  - criterion: entropy
  - splitter: best
- **Evaluation**: Holdout validation
- **Performance**:
  - **Accuracy: 0.424**
- **Strengths**: Information gain-based splits, good interpretability

### 🏅 **Rank 5: Decision Tree + Z-Score Scaling (Gini)**
- **Preprocessing**: Z-Score Scaling + Encoding
- **Model**: Decision Tree Classifier
- **Parameters**: 
  - max_depth: 3
  - criterion: gini
  - splitter: best
- **Evaluation**: Holdout validation
- **Performance**:
  - **Accuracy: 0.424**
- **Strengths**: Gini impurity-based splits, consistent performance

### 📊 **Key Insights from Modular Analysis**

#### **Preprocessing Performance**
| Scaling Method | Best Accuracy | Preferred Models |
|----------------|---------------|------------------|
| **Robust Scaling** | 0.428 | Random Forest, Decision Tree |
| **Z-Score Scaling** | 0.427 | Random Forest, Decision Tree |
| **MinMax Scaling** | - | (Lower performance) |

#### **Model Performance Ranking**
| Model Type | Best Configuration | Key Advantage |
|------------|-------------------|---------------|
| **Random Forest** | 50-100 estimators, depth 8 | **Highest accuracy** |
| **Decision Tree** | Max depth 3 | **Best interpretability** |
| **XGBoost** | (In progress) | Advanced gradient boosting |
| **Bagging** | (In progress) | Ensemble robustness |

#### **Evaluation Strategy Impact**
- **Holdout Validation**: Faster execution, good for initial screening
- **Stratified K-Fold**: More robust, better for final model selection
- **Cross-Validation**: Recommended for production deployment

### 🎯 **Selection Criteria & Framework**

Our modular analysis framework evaluates combinations based on:

1. **Preprocessing Effectiveness**: How well different scaling methods handle outliers
2. **Model Robustness**: Consistency across different evaluation strategies  
3. **Parameter Optimization**: Automated hyperparameter tuning results
4. **Clinical Applicability**: Balance between accuracy and interpretability
5. **Computational Efficiency**: Training time and resource requirements

### 🔬 **Modular Analysis Architecture**

```python
# Our unified analysis function tests combinations systematically:
comprehensive_mental_health_analysis()
├── encoding.py          # Categorical feature processing
├── scaling.py           # Numerical feature scaling  
├── evaluation_kfold.py  # Model training & validation
└── K-means.py          # Clustering analysis

# Automatically generates:
results/
├── modular_all_combinations.csv      # Complete results matrix
├── modular_top_5_combinations.csv    # Best performing models
└── modular_analysis_summary.json     # Execution metadata
```

### 💡 **Clinical Application Recommendations**

Based on our modular analysis results:

- **🏥 Clinical Decision Support**: Use **Rank 3-5 (Decision Trees)** for maximum interpretability
- **📊 Screening Applications**: Use **Rank 1-2 (Random Forest)** for highest accuracy
- **🔍 Research Studies**: Use **Rank 2** for robust cross-validation results
- **⚡ Real-time Systems**: Use **Rank 3** for fastest inference with acceptable accuracy
- **📈 Population Analysis**: Combine with clustering results for comprehensive insights

## 👥 Team & Contributions

### Team Members

| Member | Role | Primary Contributions |
|--------|------|----------------------|
| **김문기 (Moon-gi Kim)** | Lead Data Scientist & Model Architect | • Model selection and evaluation<br>• Performance analysis and interpretation<br>• Business objective definition<br>• Technical documentation |
| **김성은 (Seong-eun Kim)** | Data Engineer & Validation Specialist | • Dataset selection and acquisition<br>• K-fold cross-validation implementation<br>• Model performance evaluation<br>• Result visualization and reporting |
| **노동훈 (Dong-hun Noh)** | Feature Engineer & Scaling Specialist | • Feature type classification<br>• Robust scaling implementation<br>• Outlier detection and handling<br>• Data transformation pipeline |
| **신정현 (Jung-hyun Shin)** | Data Preprocessing & Classification Lead | • Data encoding strategies<br>• Categorical variable handling<br>• Missing data treatment<br>• Classification model implementation |
| **이지홍 (Ji-hong Lee)** | Clustering Analyst & Pattern Discovery | • K-means clustering analysis<br>• Cluster optimization and validation<br>• Pattern interpretation<br>• Unsupervised learning insights |

### Contribution Distribution
- **Data Pipeline**: 김성은, 노동훈, 신정현
- **Machine Learning**: 김문기, 신정현, 이지홍
- **Analysis & Evaluation**: 김문기, 김성은, 이지홍
- **Documentation**: All team members

### Key Files

| File | Description | Auto-Generated |
|------|-------------|----------------|
| `main_code.py` | **🎯 Main controller** - Run this for complete analysis | ❌ |
| `anxiety_depression_data.csv` | Original dataset from Kaggle | ❌ |
| `encoding.py` | Data encoding module | ❌ |
| `scaling.py` | Feature scaling module | ❌ |
| `evaluation_kfold.py` | Classification and evaluation module | ❌ |
| `K-means.py` | Clustering analysis module | ❌ |
| `results/encoded_data.csv` | Encoded categorical features | ✅ |
| `results/Robustscaling_Q1.csv` | Scaled dataset | ✅ |
| `results/kmeans_clustering_results.csv` | Clustering results | ✅ |
| `results/pipeline_execution_summary.txt` | Execution summary | ✅ |

## 🔬 Methodology

### 1. Data Preprocessing
- **Missing Value Treatment**: Replace 'None' values with meaningful categories
- **Outlier Detection**: Identify and handle extreme values using IQR method
- **Feature Encoding**: Apply appropriate encoding based on variable types
- **Data Scaling**: Use robust scaling for outlier-resistant normalization

### 2. Feature Engineering
- **Selection**: Use model-based importance scores to select top features
- **Creation**: Engineer new features like stress-to-support ratio
- **Validation**: Cross-validate feature importance across multiple models

### 3. Model Development
- **Multi-class Classification**: Initial 3-class approach (Low/Medium/High)
- **Binary Classification**: Refined approach focusing on high-risk detection
- **Ensemble Methods**: Combine multiple algorithms for robust predictions
- **Threshold Optimization**: Adjust decision boundaries for clinical needs

### 4. Evaluation Strategy
- **Cross-Validation**: 5-fold stratified cross-validation
- **Multiple Metrics**: Accuracy, precision, recall, F1-score
- **Clinical Focus**: Prioritize recall for high-risk case detection
- **Statistical Validation**: Ensure reproducible and reliable results

## 🔧 Technical Requirements

### Dependencies
```
pandas>=1.3.0          # Data manipulation and analysis
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.0.0    # Machine learning algorithms
xgboost>=1.5.0         # Gradient boosting framework
matplotlib>=3.4.0      # Basic plotting
seaborn>=0.11.0        # Statistical visualization
imbalanced-learn>=0.8.0  # Handling imbalanced datasets
jupyter>=1.0.0         # Interactive notebooks
plotly>=5.0.0          # Interactive visualizations
```

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM recommended
- **Storage**: ~100MB for dataset and models
- **OS**: Windows, macOS, or Linux

## 📚 Documentation

- **[Project Report](docs/Team10_report.md)**: Complete academic report with methodology and results
- **[API Documentation](docs/api_documentation.md)**: Function and class references  
- **[Clinical Interpretation](docs/clinical_interpretation.md)**: Healthcare implications and insights
- **[Contributing Guidelines](CONTRIBUTING.md)**: How to contribute to the project

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

