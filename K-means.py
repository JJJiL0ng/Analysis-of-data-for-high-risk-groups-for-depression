import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data (이미 적절히 스케일링된 데이터)
data = pd.read_csv("Robustscaling_Q1.csv")

# Remove index column if exists
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# Define variable categories (이미 전처리된 데이터이므로 구분)
onehot_vars = ['Gender_Female', 'Gender_Male', 'Gender_Non-Binary',
               'Employment_Status_Employed', 'Employment_Status_Retired', 
               'Employment_Status_Student', 'Employment_Status_Unemployed']

ordinal_vars = ['Education_Level', 'Medication_Use', 'Substance_Use']

numerical_vars = ['Age', 'Sleep_Hours', 'Physical_Activity_Hrs', 'Social_Support_Score',
                  'Anxiety_Score', 'Depression_Score', 'Stress_Level', 
                  'Family_History_Mental_Illness', 'Chronic_Illnesses', 'Therapy',
                  'Meditation', 'Financial_Stress', 'Work_Stress', 'Self_Esteem_Score',
                  'Life_Satisfaction_Score', 'Loneliness_Score']

print("One-hot encoded variables:", len(onehot_vars))
print("Ordinal encoded variables:", len(ordinal_vars))
print("Numerical variables (already scaled):", len(numerical_vars))

# Prepare data for clustering (중복 스케일링 제거)
def prepare_data_for_clustering(data, onehot_vars, ordinal_vars, numerical_vars):
    """이미 적절히 전처리된 데이터 사용 (중복 스케일링 방지)"""
    processed_data = data.copy()
    
    print("Using pre-processed data without additional scaling")
    print("- Numerical variables: Already scaled using Robust Scaling (Q1 method)")
    print("- One-hot encoded variables: Kept as 0/1 values")
    print("- Ordinal variables: Kept with original encoding")
    
    return processed_data, None

# Prepare data
processed_data, _ = prepare_data_for_clustering(data, onehot_vars, ordinal_vars, numerical_vars)

# 1. Find optimal number of clusters using Elbow method
def find_optimal_clusters_elbow(data, max_k=15):
    """Elbow method로 최적 클러스터 수 찾기"""
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Plot Elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker='o')
    plt.title('Finding Optimal Number of Clusters using Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.show()
    
    return inertias

# 2. Find optimal number of clusters using Silhouette score
def find_optimal_clusters_silhouette(data, max_k=15):
    """Silhouette score로 최적 클러스터 수 찾기"""
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Plot Silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title('Finding Optimal Number of Clusters using Silhouette Score')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.grid(True)
    plt.show()
    
    # Print best score and corresponding k
    best_k = k_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    print(f"Best silhouette score: {best_score:.3f} at k={best_k}")
    
    return silhouette_scores, best_k

# 3. Perform K-means clustering
def perform_kmeans_clustering(data, n_clusters):
    """K-means 클러스터링 수행"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    
    # Calculate evaluation metrics
    silhouette_avg = silhouette_score(data, cluster_labels)
    calinski_score = calinski_harabasz_score(data, cluster_labels)
    
    print(f"\nClustering Results for k={n_clusters}:")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Calinski-Harabasz Score: {calinski_score:.2f}")
    
    # Count data points in each cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("\nCluster distribution:")
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} points ({count/len(cluster_labels)*100:.1f}%)")
    
    return kmeans, cluster_labels

# 4. Visualize clusters using PCA
def visualize_clusters_pca(data, cluster_labels, n_clusters):
    """PCA를 사용한 클러스터 시각화"""
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    data_pca = pca.fit_transform(data)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_data = data_pca[cluster_labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.7)
    
    plt.title('K-means Clustering Results (PCA 2D Visualization)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pca

# 5. Analyze cluster characteristics
def analyze_cluster_characteristics(original_data, cluster_labels, 
                                 onehot_vars, ordinal_vars, numerical_vars):
    """각 클러스터의 특성 분석"""
    analysis_data = original_data.copy()
    analysis_data['Cluster'] = cluster_labels
    
    print("\n" + "="*60)
    print("CLUSTER CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    for cluster in sorted(np.unique(cluster_labels)):
        cluster_data = analysis_data[analysis_data['Cluster'] == cluster]
        print(f"\n--- CLUSTER {cluster} (n={len(cluster_data)}) ---")
        
        # One-hot encoded 변수들의 비율 분석
        if onehot_vars:
            print("\nOne-hot Encoded Variables (% of cluster):")
            for var in onehot_vars:
                if var in original_data.columns:
                    percentage = cluster_data[var].mean() * 100
                    print(f"  {var}: {percentage:.1f}%")
        
        # Ordinal 변수들의 평균/최빈값 분석
        if ordinal_vars:
            print("\nOrdinal Variables:")
            for var in ordinal_vars:
                if var in original_data.columns:
                    mean_val = cluster_data[var].mean()
                    mode_val = cluster_data[var].mode()
                    if len(mode_val) > 0:
                        print(f"  {var}: mean={mean_val:.2f}, mode={mode_val.iloc[0]}")
        
        # 수치형 변수들 분석
        if numerical_vars:
            print("\nNumerical Variables (cluster vs overall mean):")
            cluster_means = cluster_data[numerical_vars].mean()
            overall_means = original_data[numerical_vars].mean()
            
            # 전체 평균과 차이가 큰 순으로 정렬
            differences = abs(cluster_means - overall_means).sort_values(ascending=False)
            
            for var in differences.head(8).index:  # 상위 8개 변수
                cluster_val = cluster_means[var]
                overall_val = overall_means[var]
                diff = cluster_val - overall_val
                print(f"  {var}: {cluster_val:.2f} (overall: {overall_val:.2f}, diff: {diff:+.2f})")
    
    return analysis_data

# 6. Visualize key variable distributions
def plot_key_distributions(data_with_clusters, key_variables):
    """주요 변수들의 클러스터별 분포 시각화"""
    n_vars = len(key_variables)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, var in enumerate(key_variables):
        if i < len(axes) and var in data_with_clusters.columns:
            sns.boxplot(data=data_with_clusters, x='Cluster', y=var, ax=axes[i])
            axes[i].set_title(f'Distribution of {var} by Cluster')
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(key_variables), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

# Execute analysis
print("Starting K-means clustering analysis...")
print(f"Data shape: {data.shape}")

# Find optimal number of clusters
print("\n1. Finding optimal number of clusters...")
inertias = find_optimal_clusters_elbow(processed_data)
silhouette_scores, best_k = find_optimal_clusters_silhouette(processed_data)

# Perform clustering with optimal k
print(f"\n2. Performing K-means clustering with k={best_k}...")
kmeans_model, labels = perform_kmeans_clustering(processed_data, best_k)

# Visualize clusters
print("\n3. Visualizing clusters...")
pca_model = visualize_clusters_pca(processed_data, labels, best_k)

# Analyze cluster characteristics
print("\n4. Analyzing cluster characteristics...")
analysis_data = analyze_cluster_characteristics(data, labels, 
                                              onehot_vars, ordinal_vars, numerical_vars)

# Visualize key numerical variables
print("\n5. Creating distribution plots...")
key_numerical_vars = ['Anxiety_Score', 'Depression_Score', 'Stress_Level', 
                     'Sleep_Hours', 'Social_Support_Score', 'Life_Satisfaction_Score',
                     'Self_Esteem_Score', 'Loneliness_Score']

plot_key_distributions(analysis_data, key_numerical_vars)

# Save results
print("\n6. Saving results...")
result_data = data.copy()
result_data['Cluster'] = labels
result_data.to_csv('kmeans_clustering_results.csv', index=False)

print(f"\nResults saved to: kmeans_clustering_results.csv")
print(f"\nFinal clustering summary:")
print(f"- Optimal number of clusters: {best_k}")
print(f"- Silhouette score: {silhouette_score(processed_data, labels):.3f}")
print(f"- Data preprocessing: Numerical variables scaled using Robust Scaling (Q1 method)")
print(f"- One-hot encoded variables: {len(onehot_vars)}")
print(f"- Ordinal variables: {len(ordinal_vars)}")
print(f"- Numerical variables: {len(numerical_vars)}")