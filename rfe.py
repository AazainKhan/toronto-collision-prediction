import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset 
df = pd.read_csv(r"C:\Users\YounghunMun\Desktop\SupervisedLearning_274\project\KSI.csv")

print(df.info())

print(df.describe())

print(df.head())

columns_to_remove = [
    'INDEX_', 'DATE', 'NEIGHBOURHOOD_140', 'NEIGHBOURHOOD_158', 'X', 'Y', 'INITDIR', 'ObjectId', 'ACCNUM',
    'STREET1', 'STREET2', 'WARDNUM', 'DIVISION', 'OFFSET', 'HOOD_140', 'HOOD_158'
]

# Remove specified columns from the dataset
data_cleaned = df.drop(columns=columns_to_remove)

threshold = 0.5
data_cleaned = data_cleaned.dropna(axis='columns', thresh=int(threshold * len(df)))

# Handle missing values
numerical_cols = data_cleaned.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns

for col in numerical_cols:
    data_cleaned[col].fillna(data_cleaned[col].mean(), inplace=True)

for col in categorical_cols:
    data_cleaned[col].fillna(data_cleaned[col].mode()[0], inplace=True)
    

data_cleaned['ACCLASS'] = data_cleaned['ACCLASS'].astype('category').cat.codes
data_preprocessed = pd.get_dummies(data_cleaned, columns=categorical_cols.drop('ACCLASS'))

features = data_preprocessed.drop('ACCLASS', axis=1)
target = data_preprocessed['ACCLASS']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=73)

rf_model = RandomForestClassifier(n_estimators=10, random_state=73)
rfe = RFE(estimator=rf_model, n_features_to_select=10, step=1)
rfe.fit(X_train, y_train)

selected_features = features.columns[rfe.support_]
ranking = rfe.ranking_

selected_features_df = pd.DataFrame({
    'Feature': features.columns,
    'Selected': rfe.support_,
    'Ranking': rfe.ranking_
})

import seaborn as sns

# Plotting distribution graphs for the selected features using displot
# X_train instead of data_cleaned
# the selected_features of the RFE is based on processed data (features DataFrame) that contains raw encoded columns.
for feature in selected_features:
    plt.figure(figsize=(10, 6))
    sns.displot(X_train[feature].dropna(), kde=True, aspect=1.5)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density') 
    plt.show()

data_for_correlation = X_train[selected_features].copy()
data_for_correlation['ACCLASS'] = y_train

correlation_matrix = data_for_correlation.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix with ACCLASS Variable')
plt.tight_layout() # This ensures the layout fits well in the figure
plt.show()

# Show selected 10 features 
print(selected_features_df[selected_features_df['Selected'] == True])