import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImPipeline
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("../cleaned_data_KSI.csv")

# Select features and target
X = df.drop('ACCLASS', axis=1)
y = df['ACCLASS']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=5, stratify=y_encoded)

# Define transformations
numeric_features = X_train.select_dtypes(
    include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(
    include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)])

# Create a pipeline
pipeline = ImPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=5)),
    ('classifier', SVC(random_state=5, probability=True))
])

# Evaluate model before hyperparameter tuning
model = pipeline.fit(X_train, y_train)
y_pred_before = model.predict(X_test)
y_probs_before = model.predict_proba(X_test)[:, 1]
accuracy_before = accuracy_score(y_test, y_pred_before)
precision_before = precision_score(y_test, y_pred_before, average=None)
recall_before = recall_score(y_test, y_pred_before, average=None)
f1_before = f1_score(y_test, y_pred_before, average=None)
auc_roc_before = roc_auc_score(y_test, y_probs_before)
conf_matrix_before = confusion_matrix(y_test, y_pred_before)

print("Before Tuning:")
print(f"Accuracy: {accuracy_before}")
print("Precision per class:", precision_before)
print("Recall per class:", recall_before)
print("F1 Score per class:", f1_before)
print("AUC-ROC:", auc_roc_before)
print("Confusion Matrix:\n", conf_matrix_before)

# Define the parameter grid for RandomizedSearchCV
param_distributions = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'classifier__kernel': ['rbf', 'poly', 'sigmoid']
}

# Configure RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions,
                                   n_iter=10, cv=3, verbose=2, random_state=5, scoring='accuracy')
random_search.fit(X_train, y_train)

# Best model and its parameters
best_params = random_search.best_params_
best_estimator = random_search.best_estimator_
mean_cv_score = random_search.best_score_
print("After Tuning:")
print("Best Parameters:", best_params)
print("Best Estimator:", best_estimator)
print("Mean Cross Validation Score:", mean_cv_score)

# Predictions on the test set after tuning
y_pred_after = best_estimator.predict(X_test)
y_probs_after = best_estimator.predict_proba(X_test)[:, 1]
accuracy_after = accuracy_score(y_test, y_pred_after)
precision_after = precision_score(y_test, y_pred_after, average=None)
recall_after = recall_score(y_test, y_pred_after, average=None)
f1_after = f1_score(y_test, y_pred_after, average=None)
auc_roc_after = roc_auc_score(y_test, y_probs_after)
conf_matrix_after = confusion_matrix(y_test, y_pred_after)

print("After Tuning:")
print(f"Accuracy: {accuracy_after}")
print("Precision per class:", precision_after)
print("Recall per class:", recall_after)
print("F1 Score per class:", f1_after)
print("AUC-ROC:", auc_roc_after)
print("Confusion Matrix:\n", conf_matrix_after)

# Evaluate metrics for each class specifically if needed
if len(np.unique(y)) > 1:
    class_labels = label_encoder.classes_
    for index, label in enumerate(class_labels):
        print(f"{label} - Precision: {precision_after[index]}")
        print(f"{label} - Recall: {recall_after[index]}")
        print(f"{label} - F1 Score: {f1_after[index]}")

# Plot ROC curve
plt.figure()
fpr_after, tpr_after, _ = roc_curve(y_test, y_probs_after)
plt.plot(fpr_after, tpr_after, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % auc_roc_after)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic After Tuning')
plt.legend(loc="lower right")
plt.show()


