import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("anxiety_depression_data.csv")
data = data[data['Gender'] != 'Other'].reset_index(drop=True)  # 'Other' 성별은 제외

# 점수를 기반으로 우울 수준을 세 범주로 나누는 함수 정의
def get_depression_level(score):
    if score >= 15:
        return "High"
    elif score >= 10:
        return "Medium"
    else:
        return "Low"

# 새 컬럼으로 우울 수준을 추가
data['Depression_Level'] = data['Depression_Score'].apply(get_depression_level)

# 결측값 처리
data['Medication_Use'] = data['Medication_Use'].replace(np.nan, 'No')
data['Substance_Use'] = data['Substance_Use'].replace(np.nan, 'No')

# 순서형 컬럼 지정
ordinal_cols = ['Education_Level', 'Medication_Use', 'Substance_Use', 'Depression_Level']
ordinal_categories = [
    ['Other', 'High School', "Bachelor's", "Master's", 'PhD'],
    ['No', 'Occasional', 'Regular'],
    ['No', 'Occasional', 'Frequent'],
    ['Low', 'Medium', 'High']
]

# 순서형 인코딩
ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
ordinal_encoded = ordinal_encoder.fit_transform(data[ordinal_cols])
ordinal_df = pd.DataFrame(ordinal_encoded, columns=ordinal_cols)

# 원-핫 인코딩 대상 컬럼
one_hot_cols = ['Gender', 'Employment_Status']
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(data[one_hot_cols])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))

# 인코딩하지 않은 나머지 수치형 컬럼 추출
non_encoded_cols = data.drop(columns=one_hot_cols + ordinal_cols).reset_index(drop=True)

# 최종 데이터프레임 구성
final_df = pd.concat([non_encoded_cols, one_hot_df, ordinal_df], axis=1)

# Function to test various model + scaler + evaluation combinations
def combination_selection(dataframe):
    # Define scalers to normalize the data
    scalers = {
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'zscore': StandardScaler()
    }
    # Define classifiers and parameter variations
    models = {
        'DecisionTree': [
            DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42),
            DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42 ),
            DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42),
            DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42 )
        ],
        'RandomForest': [
            RandomForestClassifier(n_estimators=50,max_depth=8,max_features='sqrt',random_state=42),
            RandomForestClassifier(n_estimators=100,max_depth=8,max_features='log2',random_state=42),
            RandomForestClassifier(n_estimators=100,max_depth=12,max_features='sqrt',random_state=42),
            RandomForestClassifier(n_estimators=100,max_depth=12,max_features='log2',random_state=42)
        ]
    }
    # Define evaluation methods
    def holdout_evaluation(clf, Xs, ys):
        X_train, X_test = Xs
        y_train, y_test = ys
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def kfold_evaluation(clf, Xs, ys):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(clf, Xs, ys, cv=kf).mean()

    def stratified_kfold_evaluation(clf, Xs, ys):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(clf, Xs, ys, cv=skf).mean()

    evaluation_methods = {
        'holdout': holdout_evaluation,
        'kfold': kfold_evaluation,
        'stratifiedkfold': stratified_kfold_evaluation
    }

    # Split features and target
    X = dataframe.drop(columns=['Depression_Level', 'Depression_Score'])
    y = dataframe['Depression_Level'].astype(int) 

    results = []
    # Loop over each scaler
    for scaler_name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        # Loop over each model and its parameter variations
        for model_name, model_list in models.items():
            for model in model_list:
                # Loop over each evaluation method
                for eval_name, eval_func in evaluation_methods.items():
                    if eval_name == 'holdout':
                        # Use train/test split for holdout evaluation
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=0.2, stratify=y, random_state=42)
                        score = eval_func(model, (X_train, X_test), (y_train, y_test))
                    else:
                        # Use cross-validation for k-fold methods
                        score = eval_func(model, X_scaled, y)
                    # Store result
                    results.append({
                        'Scaler': scaler_name,
                        'Model': type(model).__name__,
                        'Params': model.get_params(),
                        'Evaluation': eval_name,
                        'Accuracy': score
                    })
    # Convert results to DataFrame and sort by best accuracy
    result_df = pd.DataFrame(results)
    top5 = result_df.sort_values(by='Accuracy', ascending=False).head(5).reset_index(drop=True)
    # Print top 5 and best combination 
    print("Top 5 and best combination:\n")
    for i, row in top5.iterrows():
        print(f"[{i+1}] Scaler: {row['Scaler']}")
        print(f"    Model: {row['Model']}")
        print(f"    Evaluation: {row['Evaluation']}")
        print(f"    Accuracy: {row['Accuracy']}")
        print("    Params:")
        for key, val in row['Params'].items():
            print(f"        {key}: {val}")
        print()

    return top5
# Run the function and get top 5 performing combinations
top5_results = combination_selection(final_df)
