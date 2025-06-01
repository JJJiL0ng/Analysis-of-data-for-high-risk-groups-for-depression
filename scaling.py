import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, MinMaxScaler

# Load data with correct filename
data = pd.read_csv("anxiety_depression_data.csv")

# Remove rows where 'Gender' is 'Other'
data = data[data['Gender'] != 'Other'].reset_index(drop=True)

# Replace NaN values in 'Medication_Use' and 'Substance_Use' with 'No'
# (NaN does not mean missing in this context)
data['Medication_Use'] = data['Medication_Use'].replace(np.nan, 'No')
data['Substance_Use'] = data['Substance_Use'].replace(np.nan, 'No')

# Define ordinal columns and their order
ordinal_cols = ['Education_Level', 'Medication_Use', 'Substance_Use']

ordinal_categories = [
    ['Other', 'High School', "Bachelor's", "Master's", 'PhD'],   # Education_Level
    ['No', 'Occasional', 'Regular'],                               # Medication_Use
    ['No', 'Occasional', 'Frequent']                               # Substance_Use             
]

ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
ordinal_encoded = ordinal_encoder.fit_transform(data[ordinal_cols])
ordinal_df = pd.DataFrame(ordinal_encoded, columns=ordinal_cols)

# Define columns for one-hot encoding
one_hot_cols = ['Gender', 'Employment_Status']
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(data[one_hot_cols])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))

# Define categorical (binary) columns to exclude from scaling
exclude_scaling_cols = [
    'Family_History_Mental_Illness', 
    'Chronic_Illnesses', 
    'Therapy', 
    'Meditation'
]

exclude_scaling_df = data[exclude_scaling_cols].reset_index(drop=True)

# Identify numeric columns that need scaling (exclude encoded and binary columns)
encoded_cols = one_hot_cols + ordinal_cols + exclude_scaling_cols
numeric_cols = [
    col for col in data.select_dtypes(include=[np.number]).columns
    if col not in exclude_scaling_cols
]
numeric_df = data[numeric_cols].reset_index(drop=True)

# Define robust scaling function using interquartile range
# x' = (x - Q1) / (Q3 - Q1), robust to outliers
def robust_scaling_q1(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    iqr = iqr.replace(0, 1)  # Avoid division by zero
    return (df - q1) / iqr

# Apply robust scaling to numeric data
scaled_numeric_df = robust_scaling_q1(numeric_df)

# (Optional) Alternative scaling methods:
# Robust Scaling using sklearn
#scaler = RobustScaler()
#scaled_array = scaler.fit_transform(numeric_df)

#Z-score Scaling
#scaler = StandardScaler()
#scaled_array = scaler.fit_transform(numeric_df)

#min-max scaling
#scaler = MinMaxScaler()
#scaled_array = scaler.fit_transform(numeric_df)

# Convert scaled array to DataFrame (if using sklearn scalers above)
#scaled_numeric_df = pd.DataFrame(scaled_array, columns=numeric_df.columns)

# Concatenate all transformed data into the final DataFrame
final_df = pd.concat([scaled_numeric_df, exclude_scaling_df, one_hot_df, ordinal_df], axis=1)

# Display first 10 rows of the final DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(final_df.head(10))

# Save the processed data to CSV file (FIXED: removed problematic path and uncommented)
final_df.to_csv('Robustscaling_Q1.csv', index=False)
print(f"\n✅ Scaled data saved to: Robustscaling_Q1.csv")
print(f"✅ Final dataset shape: {final_df.shape}")