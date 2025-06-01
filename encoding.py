import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load data with correct filename
data = pd.read_csv("anxiety_depression_data.csv")

# Remove 'Other' from 'Gender' -> Remove 22 rows
data = data[data['Gender'] != 'Other'].reset_index(drop=True)

# Change to 'None' → 'No' (None's meaning is not a missing value)
data['Medication_Use'] = data['Medication_Use'].replace('None', 'No')
data['Medication_Use'] = data['Medication_Use'].replace('None', 'No').fillna('No')
data['Substance_Use'] = data['Substance_Use'].replace('None', 'No')
data['Substance_Use'] = data['Substance_Use'].replace('None', 'No').fillna('No')

# ordinal encoding
ordinal_cols = ['Education_Level', 'Medication_Use', 'Substance_Use']

ordinal_categories = [
    ['Other', 'High School', "Bachelor's", "Master's", 'PhD'],     # Education_Level
    ['No', 'Occasional', 'Regular'],                               # Medication_Use
    ['No', 'Occasional', 'Frequent']                               # Substance_Use             
]

ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
ordinal_encoded = ordinal_encoder.fit_transform(data[ordinal_cols])
ordinal_df = pd.DataFrame(ordinal_encoded, columns=ordinal_cols)

# one-hot encoding
one_hot_cols = ['Gender', 'Employment_Status']
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(data[one_hot_cols])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))

final_df = pd.concat([one_hot_df, ordinal_df], axis=1)

# Output only from end to row 10
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(final_df.tail(10))

# Save encoded data
final_df.to_csv('encoded_data.csv', index=False)
print(f"\n✅ Encoded data saved to: encoded_data.csv")
print(f"✅ Encoded dataset shape: {final_df.shape}")