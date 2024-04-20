
from joblib import dump
import scikitplot as skplt
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Load the data
df_group5 = pd.read_csv('../cleaned_data_KSI.csv')


print("Data Information:")
df_group5.info()


print("\nDescriptive Statistics:")
print(df_group5.describe().T)


# Separate features and target variable
x_group5 = df_group5.drop(columns=['ACCLASS'], axis=1)
y_group5 = df_group5['ACCLASS']

# Stratified split for training and testing data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(x_group5, y_group5):
    x_train, x_test = x_group5.loc[train_index], x_group5.loc[test_index]
    y_train, y_test = y_group5.loc[train_index], y_group5.loc[test_index]

# Identify numerical and categorical features
numerical_cols = x_train.select_dtypes(include=np.number).columns
cat_cols = x_train.select_dtypes(include='object').columns

# Data preprocessing pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, cat_cols)
])

log_reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
])


# null values after imputation
print("\nNumber of null values after imputation:")
print(x_train.isnull().sum())


# Prepare training data
x_train_prepared = log_reg_pipeline.fit_transform(x_train)

# Apply SMOTE for class imbalance handling
x_train_prepared, y_train = SMOTE(
    random_state=42).fit_resample(x_train_prepared, y_train)

# Transform testing data using the same pipeline
x_test_transformed = log_reg_pipeline.transform(x_test)


# Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train_prepared, y_train)

# Performance Evaluation - Before Tuning

# Cross-validation
crossvalscore = KFold(n_splits=10, random_state=42, shuffle=True)
scores = cross_val_score(log_reg, x_train_prepared,
                         y_train, cv=crossvalscore, scoring='accuracy')
print("\nCross Validation Scores (10 folds): ", scores)
print("Mean Cross Validation Score: ", scores.mean())

# Accuracy on training data
y_pred = log_reg.predict(x_train_prepared)
accuracy_train = accuracy_score(y_train, y_pred)
print("\nTraining Accuracy: ", accuracy_train)

# Accuracy on testing data
y_pred_test_before = log_reg.predict(x_test_transformed)
accuracy_test_before = accuracy_score(y_test, y_pred_test_before)
print("\nTesting Accuracy (Before Tuning): ", accuracy_test_before)

# Confusion Matrix (Before Tuning)
cfs_matrix_test_before = confusion_matrix(
    y_test, y_pred_test_before, labels=log_reg.classes_)
print("\nConfusion Matrix (Before Tuning):")
print(cfs_matrix_test_before)

# Classification Report (Before Tuning)
report = classification_report(y_train, y_pred)
print("\nClassification Report (Before Tuning):")
print(report)

# ROC Curve (Before Tuning)
y_probas = log_reg.predict_proba(x_test_transformed)
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.title('ROC Curves of Logistic Regression model (Before Tuning)')
plt.show()


# Define hyperparameter grid
param_grid = {
    'C': np.logspace(-4, 4, 50),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300, 400, 500]
}

# Randomized Search CV - Perform Grid Search for best parameters
grid = RandomizedSearchCV(log_reg, param_grid, cv=5, n_jobs=-1,
                          scoring='accuracy', n_iter=100, random_state=5)
grid.fit(x_train_prepared, y_train)

print("\nBest parameters: ", grid.best_params_)
print("\nBest estimator: ", grid.best_estimator_)
print("Best cross-validation score: ", grid.best_score_)


# Logistic Regression model with best parameters
log_reg_best = grid.best_estimator_

# Performance Evaluation - After Tuning

# Accuracy on training data
y_pred_train_after = log_reg_best.predict(x_train_prepared)
accuracy_train_after = accuracy_score(y_train, y_pred_train_after)
print("\nTraining Accuracy (After Tuning): ", accuracy_train_after)

# Accuracy on testing data
y_pred = log_reg_best.predict(x_test_transformed)
accuracy_test = accuracy_score(y_test, y_pred)
print("\nTesting Accuracy (After Tuning): ", accuracy_test)

# Classification Report (After Tuning)
report = classification_report(y_test, y_pred)
print("\nClassification Report (After Tuning):")
print(report)

# ROC Curve (After Tuning)
y_probas = log_reg_best.predict_proba(x_test_transformed)
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.title('ROC Curves of Logistic Regression model (After Tuning)')
plt.show()

# Confusion Matrix (After Tuning)
cfs_matrix = confusion_matrix(y_test, y_pred, labels=log_reg_best.classes_)
display = ConfusionMatrixDisplay(
    cfs_matrix, display_labels=log_reg_best.classes_)
display.plot()
plt.title("Confusion Matrix after tuning model")
plt.show()


# confusion matrix (after tuning)
print("Confusion Matrix:")
print(cfs_matrix)


# Save the tuned model for deployment
dump(log_reg_best, '../deployment/log_reg_model.pkl')

# Save the preprocessing pipeline for future data transformation
dump(log_reg_pipeline, '../deployment/log_reg_pipeline.pkl')

print("\nModel and Pipeline successfully dumped for deployment!")


