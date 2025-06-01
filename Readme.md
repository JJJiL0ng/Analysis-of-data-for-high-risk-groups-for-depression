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
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mental-health-prediction.git
   cd mental-health-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   ```bash
   # Place anxiety_depression_data.csv in data/raw/ directory
   mkdir -p data/raw
   # Download from Kaggle and place in data/raw/
   ```

### Usage

1. **Run data preprocessing**:
   ```bash
   python src/data_preprocessing.py
   ```

2. **Train classification models**:
   ```bash
   python src/models/classification.py
   ```

3. **Perform clustering analysis**:
   ```bash
   python src/models/clustering.py
   ```

4. **Generate evaluation reports**:
   ```bash
   python src/models/evaluation.py
   ```

## 🏗️ Architecture

The project follows a modular architecture designed for scalability and maintainability:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  Jupyter Notebooks  │  Visualization  │  Reports & Results │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  Classification Models  │  Clustering Analysis │ Evaluation │
│  • Decision Tree        │  • K-means Clustering│ • Cross-   │
│  • Random Forest        │  • Silhouette Score  │   validation│
│  • XGBoost             │  • Elbow Method      │ • Metrics  │
│  • Bagging Classifier  │                      │ • Reports  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  Data Preprocessing  │  Feature Engineering │ Model Training│
│  • Data Cleaning     │  • Feature Selection │ • Hyperparameter│
│  • Encoding          │  • Scaling          │   Tuning     │
│  • Outlier Handling  │  • SMOTE/SMOTENC    │ • Threshold  │
│                      │                     │   Optimization│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
├─────────────────────────────────────────────────────────────┤
│    Raw Data Storage    │    Processed Data    │  Model Storage│
│  • anxiety_depression  │  • Encoded Features  │ • Trained    │
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

## 📈 Model Performance

### Classification Results (Binary: High Risk vs. Not High Risk)

| Model | Accuracy | Precision (High Risk) | Recall (High Risk) | F1-Score |
|-------|----------|----------------------|-------------------|----------|
| Random Forest | 0.69 | 0.65 | 0.58 | 0.61 |
| Decision Tree | 0.59 | 0.52 | 0.71 | 0.60 |
| XGBoost* | 0.43 | 0.38 | 0.85 | 0.52 |
| Bagging* | 0.32 | 0.31 | 0.92 | 0.46 |

*Models optimized for high recall to maximize detection of high-risk cases

### Clustering Analysis
- **Optimal Clusters**: 2 clusters identified using silhouette analysis
- **Cluster 0 (60-65%)**: Mental Health Stable Group
  - Low anxiety, depression, and stress levels
  - High self-esteem scores
  - Stable sleep patterns
- **Cluster 1 (35-40%)**: Mental Health High-Risk Group
  - High anxiety, depression, and stress levels
  - Low self-esteem scores
  - Irregular sleep patterns

### Key Predictive Features
1. **Anxiety Score** - Strongest predictor of depression risk
2. **Self-Esteem Score** - Strong inverse correlation with depression
3. **Stress Level** - Major contributing factor
4. **Sleep Hours** - Sleep quality impacts mental health
5. **Life Satisfaction Score** - Overall well-being indicator

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

- **[Methodology](docs/methodology.md)**: Detailed technical approach
- **[API Documentation](docs/api_documentation.md)**: Function and class references
- **[Clinical Interpretation](docs/clinical_interpretation.md)**: Healthcare implications
- **[Contributing Guidelines](CONTRIBUTING.md)**: How to contribute to the project

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting
