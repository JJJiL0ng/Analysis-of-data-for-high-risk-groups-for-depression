import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from imblearn.over_sampling import SMOTENC

data = pd.read_csv("anxiety_depression_data.csv")
data = data[data['Gender'] != 'Other'].reset_index(drop=True)

# Low: 0~9
# Medium: 10~14
# High: 15 이상

def get_depression_level(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Medium"
    else:
        return "Low"

data['Depression_Level'] = data['Depression_Score'].apply(get_depression_level)

data['Medication_Use'] = data['Medication_Use'].replace(np.nan, 'No')
data['Substance_Use'] = data['Substance_Use'].replace(np.nan, 'No')

ordinal_cols = ['Education_Level', 'Medication_Use', 'Substance_Use', 'Depression_Level']
ordinal_categories = [
    ['Other', 'High School', "Bachelor's", "Master's", 'PhD'],   # Education_Level
    ['No', 'Occasional', 'Regular'],                               # Medication_Use
    ['No', 'Occasional', 'Frequent'],                               # Substance_Use
    ['Low', 'Medium', 'High']                                       # Depression_Level
]
ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
ordinal_encoded = ordinal_encoder.fit_transform(data[ordinal_cols])
ordinal_df = pd.DataFrame(ordinal_encoded, columns=ordinal_cols)

one_hot_cols = ['Gender', 'Employment_Status']
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(data[one_hot_cols])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))

# 인코딩 컬럼 제외한 원본 데이터 추출
non_encoded_cols = data.drop(columns=one_hot_cols + ordinal_cols).reset_index(drop=True)

# 인코딩된 컬럼과 원본의 나머지 컬럼을 합치기
final_df = pd.concat([non_encoded_cols, one_hot_df, ordinal_df], axis=1)

# Robust Scaling only on numeric columns
scaler = RobustScaler()
numeric_cols = non_encoded_cols.select_dtypes(include=[np.number]).columns
scaled_numeric = scaler.fit_transform(non_encoded_cols[numeric_cols])
scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_cols)

# Keep non-numeric columns as they are
non_numeric_df = non_encoded_cols.drop(columns=numeric_cols).reset_index(drop=True)

# Reassemble the final DataFrame
final_df = pd.concat([scaled_numeric_df, non_numeric_df, one_hot_df, ordinal_df], axis=1)

# Construct the final scaled DataFrame
scaled_df = final_df

# 1. X, y 정의
X = scaled_df.drop(columns=['Depression_Level', 'Depression_Score'])  # 타깃 제외
y = scaled_df['Depression_Level']  # 원본에서 타깃 추출

# 2. 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Decision Tree 학습
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred_dt = dt_model.predict(X_test)
print(classification_report(y_test, y_pred_dt))

# Decision Tree - Feature Importance 시각화
feature_importance = pd.Series(dt_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance from Decision Tree')
plt.show()

#1. RandomForest 분류기
rf_model = RandomForestClassifier(random_state=42)
#2. 학습
rf_model.fit(X_train, y_train)
#3. 예측 및 평가
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Random Forest - Feature Importance 시각화
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance from Random Forest')
plt.show()

# 1. XGBoost 분류기
xgb_model = XGBClassifier(random_state=42)

# 2. 학습
xgb_model.fit(X_train, y_train)

# 3. 예측 및 평가
y_pred_xgb = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred_xgb))

# plot_importance Visualizes the importance of each feature
plot_importance(xgb_model, importance_type='gain') 
plt.show()

# 선택된 중요 feature
selected_features = [
    'Age', 'Physical_Activity_Hrs', 'Sleep_Hours',
    'Anxiety_Score', 'Life_Satisfaction_Score', 'Social_Support_Score',
    'Stress_Level', 'Work_Stress', 'Loneliness_Score', 'Therapy',
    'Financial_Stress'
]

# 해당 feature만 추출
X_selected = scaled_df[selected_features]
y = scaled_df["Depression_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
print(classification_report(y_test, y_pred_dt))

#2. 학습
rf_model.fit(X_train, y_train)
#3. 예측 및 평가
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

xgb_model.fit(X_train, y_train)

# 3. 예측 및 평가
y_pred_xgb = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred_xgb))

# 기존 Depression_Level에서, 'High'만 1로, 나머지는 0으로 설정
binary_y = (scaled_df['Depression_Level'] == 2).astype(int)

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, binary_y, test_size=0.2, random_state=42, stratify=binary_y)

# 3. Decision Tree 학습
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_model.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree report")
print(classification_report(y_test, y_pred_dt))

#1. RandomForest 분류기
rf_model = RandomForestClassifier(random_state=42)
#2. 학습
rf_model.fit(X_train, y_train)
#3. 예측 및 평가
y_pred_rf = rf_model.predict(X_test)
print("Random Forest report")
print(classification_report(y_test, y_pred_rf))

# 1. XGBoost 분류기
xgb_model = XGBClassifier(random_state=42)

# 2. 학습
xgb_model.fit(X_train, y_train)

# 3. 예측 및 평가
y_pred_xgb = xgb_model.predict(X_test)

bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    n_estimators=200,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

bag_model.fit(X_train, y_train)
y_pred_bag = bag_model.predict(X_test)
print("BaggingClassifier report")
print(classification_report(y_test, y_pred_bag))

# ordinal feature 인덱스 구하기
ordinal_feature_idx = [X.columns.get_loc(col) for col in ordinal_cols if col in X.columns]

# SMOTENC 적용
X_train, X_test, y_train, y_test = train_test_split(X, binary_y, test_size=0.2, stratify=binary_y, random_state=42)
sm = SMOTENC(categorical_features=ordinal_feature_idx, random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
# 3. Deicision Tree 학습
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_model.fit(X_resampled, y_resampled)

# 4. 예측 및 평가
y_pred_dt = dt_model.predict(X_test)
print(classification_report(y_test, y_pred_dt))

# Decision Tree - Feature Importance 시각화
feature_importance = pd.Series(dt_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance from Decision Tree')
plt.show()

#1. RandomForest 분류기
rf_model = RandomForestClassifier(random_state=42)
#2. 학습
rf_model.fit(X_resampled, y_resampled)
#3. 예측 및 평가
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Random Forest - Feature Importance 시각화
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance from Random Forest')
plt.show()

new_features = pd.DataFrame({
    'physical_per_awake_hour': data['Physical_Activity_Hrs'] / (24 - data['Sleep_Hours']),
    'stress_support_ratio': data['Stress_Level'] / data['Social_Support_Score'],
    'isolation_index': data['Loneliness_Score'] / data['Social_Support_Score']
})

X = scaled_df.drop(columns=['Depression_Level', 'Depression_Score'])  # 타깃 제외  
X_train, X_test, y_train, y_test = train_test_split(
    X, binary_y, test_size=0.2, random_state=42, stratify=binary_y
)

# Bagging 모델로 확률 예측
# predict_proba returns the probability of each class 0, 1 for every sample
# [:, 1] extracts the probability of the positive class (class 1)
y_probs = bag_model.predict_proba(X_test)[:, 1]

# 여러 threshold에 대한 precision-recall 계산
# precision_recall_curve compares the true labels (y_test) with the predicted probabilities (y_probs)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# 테이블 생성
threshold_table = pd.DataFrame({
    'threshold': np.append(thresholds, 1.0),
    'precision': precisions,
    'recall': recalls
})

# PR 그래프
plt.figure(figsize=(8,6))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold (BaggingClassifier)')
plt.legend()
plt.grid(True)
plt.show()

# 1. XGBoost 분류기
xgb_model = XGBClassifier(random_state=42, scale_pos_weight=2.0)

# 2. 학습
xgb_model.fit(X_train, y_train)

# 3. 예측 및 평가
y_pred_xgb = xgb_model.predict(X_test)

# 확률 예측
y_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Precision-Recall 계산
precisions_xgb, recalls_xgb, thresholds_xgb = precision_recall_curve(y_test, y_probs_xgb)

# 테이블 생성
threshold_table_xgb = pd.DataFrame({
    'threshold': np.append(thresholds_xgb, 1.0),
    'precision': precisions_xgb,
    'recall': recalls_xgb
})

# PR 그래프
plt.figure(figsize=(8, 6))
plt.plot(thresholds_xgb, precisions_xgb[:-1], label='Precision')
plt.plot(thresholds_xgb, recalls_xgb[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold (XGBoost)')
plt.legend()
plt.grid(True)
plt.show()

# Threshold를 위한 validation set 분리
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(X, binary_y, test_size=0.2, stratify=binary_y, random_state=42)

# XGBoost 모델 훈련 및 threshold tuning
# scale_pos_weight is used to give more weight to the positive class
xgb_model = XGBClassifier(random_state=42, scale_pos_weight=2.0)
xgb_model.fit(X_train_val, y_train_val)
val_probs_xgb = xgb_model.predict_proba(X_holdout)[:, 1]
# Compute precision, recall, and thresholds for different cutoff values using the holdout labels
prec_xgb, rec_xgb, thr_xgb = precision_recall_curve(y_holdout, val_probs_xgb)
high_recall = rec_xgb[1:] >= 0.7
# Select the threshold that gives the highest precision
# If no threshold satisfies the recall condition, default to 0.5
best_thr_xgb = thr_xgb[high_recall][np.argmax(prec_xgb[1:][high_recall])] if high_recall.any() else 0.5


# Bagging 모델 훈련 및 threshold tuning
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    n_estimators=200,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
bag_model.fit(X_train_val, y_train_val)
# Get predicted probabilities for the positive class on the holdout set
val_probs_bag = bag_model.predict_proba(X_holdout)[:, 1]
# Use the previously selected best threshold to generate final predictions
y_pred_best_xgb = (val_probs_xgb >= best_thr_xgb).astype(int)
prec_bag, rec_bag, thr_bag = precision_recall_curve(y_holdout, val_probs_bag)
# From thresholds with recall >= 0.7, choose the one with the highest precision
high_recall = rec_bag[1:] >= 0.7
best_thr_bag = thr_bag[high_recall][np.argmax(prec_bag[1:][high_recall])] if high_recall.any() else 0.5
# Generate final predictions for the bagging model using the best threshold
y_pred_best_bag = (val_probs_bag >= best_thr_bag).astype(int)

print("=== Classification Report @ Best Threshold (XGBoost) ===")
print(classification_report(y_holdout, y_pred_best_xgb))
print(f"[XGBoost best threshold] {best_thr_xgb:.2f}")

print("=== Classification Report @ Best Threshold (Bagging) ===")
print(classification_report(y_holdout, y_pred_best_bag))
print(f"[Bagging best threshold] {best_thr_bag:.2f}")

# k-fold cross validation for testing classification models
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store accuracy scores for each model across folds
dt_scores, rf_scores, xgb_scores, bag_scores = [], [], [], []

print("\n=== k-fold cross validation ===")

# Perform 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kfold.split(X, binary_y), start=1):
    print(f"\n--------------------- Fold {fold} ---------------------")
    # Split the data into training and test sets
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = binary_y.iloc[train_idx], binary_y.iloc[test_idx]

    # Decision Tree
    dt_model.fit(X_tr,  y_tr)
    dt_preds = dt_model.predict(X_te)
    # Store the accuracy score for current fold
    dt_scores.append(dt_model.score(X_te, y_te))
    # Print evaluation results of the Decision Tree
    print("\n--- Decision Tree ---")
    print("\nConfusion Matrix:\n", confusion_matrix(y_te, dt_preds))
    print("Classification Report:\n", classification_report(y_te, dt_preds))

    # Random Forest
    rf_model.fit(X_tr,  y_tr)
    rf_preds = rf_model.predict(X_te)
    # Store the accuracy score for current fold
    rf_scores.append(rf_model.score( X_te, y_te))
    # Print evaluation results of the Random Forest
    print("--- Random Forest ---")
    print("\nConfusion Matrix:\n", confusion_matrix(y_te, rf_preds))
    print("Classification Report:\n", classification_report(y_te, rf_preds))

    # XGBoost
    xgb_model.fit(X_tr, y_tr)
    # Predict probabilities for class 1 on the test set
    probs_xgb = xgb_model.predict_proba(X_te)[:, 1]
    # If the probability >= threshold, then predict 1; otherwise, predict 0
    y_pred_xgb = (probs_xgb >= best_thr_xgb).astype(int)
    # Check if the prediction is correct
    # Store the accuracy score for current fold
    acc_xgb = (y_pred_xgb == y_te).mean()
    xgb_scores.append(float(acc_xgb))
    # Print evaluation results of the XGBoost
    print("--- XGBoost ---")
    print("\nConfusion Matrix:\n", confusion_matrix(y_te, y_pred_xgb))
    print("Classification Report:\n", classification_report(y_te, y_pred_xgb))

    # Bagging
    bag_model.fit(X_tr, y_tr)
    # Predict probabilities for class 1 on the test set
    probs_bag = bag_model.predict_proba(X_te)[:, 1]
    # Apply a threshold to convert probabilities to binary class labels
    y_pred_bag = (probs_bag >= best_thr_bag).astype(int)
    # Check if the prediction is correct
    # Store the accuracy score for current fold
    acc_bag = (y_pred_bag == y_te).mean()
    bag_scores.append(float(acc_bag))
    # Print evaluation results of the Bagging 
    print("--- Bagging ---")
    print("\nConfusion Matrix:\n", confusion_matrix(y_te, rf_preds))
    print("Classification Report:\n", classification_report(y_te, y_pred_bag))

# Print the accuracy of the models
print("\n==== Accuracy ====")
print("\n--- Decision Tree ---")
print("Accuracy for each fold :", dt_scores)
print("Average accuracy:", np.mean(dt_scores))
print("\n--- Random Forest ---")
print("Accuracy for each fold:", rf_scores)
print("Average accuracy:", np.mean(rf_scores))
print("\n--- XGBoost ---")
print("Accuracy for each fold:", xgb_scores)
print("Average accuracy:", np.mean(xgb_scores))
print("\n--- Bagging ---")
print("Accuracy for each fold:", bag_scores)
print("Average accuracy:", np.mean(bag_scores))

# Plot accuracy per fold for each model
plt.plot(range(1, 6), dt_scores, marker='o', label='Decision Tree')
plt.plot(range(1, 6), rf_scores, marker='o', label='Random Forest')
plt.plot(range(1, 6), xgb_scores, marker='o', label='XGBoost (Threshold)')
plt.plot(range(1, 6), bag_scores, marker='o', label='Bagging (Threshold)')
plt.title('Accuracy per Fold with Fixed Threshold')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()