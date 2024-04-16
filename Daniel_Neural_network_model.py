# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_imb  # Using imbalanced-learn's pipeline

# Load the cleaned dataset
df = pd.read_csv('cleaned_data_KSI.csv')
print(df.info())

# Separating the features and target variable
X = df.drop(columns=['ACCLASS'], axis=1)
y = df['ACCLASS']

# Setting up the train-test split with stratified shuffle to maintain distribution
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Define numerical and categorical pipelines
numerical_cols = X_train.select_dtypes(include=np.number).columns
categorical_cols = X_train.select_dtypes(include='object').columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine numerical and categorical pipelines into a preprocessor
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

# Define the full pipeline including the preprocessor and the MLP classifier
mlp_pipeline = make_pipeline_imb(
    preprocessor,
    SMOTE(random_state=42),
    MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300, random_state=42)
)

# Train the neural network
mlp_pipeline.fit(X_train, y_train)

# Predict and evaluate the model
y_train_pred = mlp_pipeline.predict(X_train)
y_test_pred = mlp_pipeline.predict(X_test)

print("Training Accuracy: ", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy: ", accuracy_score(y_test, y_test_pred))
print("Recall Score: ", recall_score(y_test, y_test_pred, pos_label='Fatal'))
print("Precision Score: ", precision_score(y_test, y_test_pred, pos_label='Fatal'))
print("Classification Report:\n", classification_report(y_test, y_test_pred))
