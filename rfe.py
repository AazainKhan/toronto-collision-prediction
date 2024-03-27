import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# Load the dataset 
df = pd.read_csv(r"C:\Users\YounghunMun\Desktop\SupervisedLearning_274\project\KSI.csv")

columns_to_remove = [
    'INDEX_', 'DATE', 'NEIGHBOURHOOD_140', 'NEIGHBOURHOOD_158', 'X', 'Y', 'INITDIR',
    'STREET1', 'STREET2', 'WARDNUM', 'DIVISION'
]

# Remove specified columns from the dataset
data_cleaned = df.drop(columns=columns_to_remove)

# Handle missing values
numerical_cols = data_cleaned.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns

for col in numerical_cols:
    data_cleaned[col].fillna(data_cleaned[col].mean(), inplace=True)

for col in categorical_cols:
    data_cleaned[col].fillna(data_cleaned[col].mode()[0], inplace=True)

data_preprocessed = pd.get_dummies(data_cleaned, columns=categorical_cols.drop('ACCLASS'))

target = data_cleaned['ACCLASS']
features = data_preprocessed.drop('ACCLASS', axis=1)

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=10, random_state=73)

rfe = RFE(estimator=rf_model, n_features_to_select=10, step=1)

# Fit RFE
rfe.fit(X_train, y_train)

selected_features = features.columns[rfe.support_]
ranking = rfe.ranking_

selected_features_df = pd.DataFrame({
    'Feature': features.columns,
    'Selected': rfe.support_,
    'Ranking': rfe.ranking_
})

# Show selected 10 features 
print(selected_features_df[selected_features_df['Selected'] == True])
