# Import libraries
from joblib import dump
import scikitplot as skplt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Load the dataset
df_official = pd.read_csv('../cleaned_data_KSI.csv')


# Display dataset information
print(df_official.info())


# Display dataset statistics
print(df_official.describe().T)


# Separate features and target variable
x_group3 = df_official.drop(columns=['ACCLASS'], axis=1)
y_group3 = df_official['ACCLASS']


# Split the dataset into training and testing sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(x_group3, y_group3):
    x_train, x_test = x_group3.loc[train_index], x_group3.loc[test_index]
    y_train, y_test = y_group3.loc[train_index], y_group3.loc[test_index]


# Print class distribution before using SMOTE
print("Before using SMOTE")
print(y_train.value_counts())


# Preprocessing pipelines for numerical and categorical features
numerical_cols = x_train.select_dtypes(include=np.number).columns
cat_cols = x_train.select_dtypes(include='object').columns

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

svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

# Transform training data and apply SMOTE
x_train_prepared = svm_pipeline.fit_transform(x_train)
x_train_prepared, y_train = SMOTE(
    random_state=42).fit_resample(x_train_prepared, y_train)

# Print class distribution after using SMOTE
print("After using SMOTE")
print(y_train.value_counts())


# Train SVM model
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(x_train_prepared, y_train)

# Evaluate the model
crossvalscore = KFold(n_splits=10, random_state=42, shuffle=True)
scores = cross_val_score(svm_model, x_train_prepared,
                         y_train, cv=crossvalscore, scoring='accuracy')
print("Cross validation scores for 10 folds: ", scores)
print("Mean cross validation score: ", scores.mean())


# Make predictions on training data
y_pred_train = svm_model.predict(x_train_prepared)
print("Accuracy score on training data: ",
      accuracy_score(y_train, y_pred_train))


# Make predictions on testing data
x_test_transformed = svm_pipeline.transform(x_test)
y_pred_test = svm_model.predict(x_test_transformed)
print("Accuracy score on testing data: ", accuracy_score(y_test, y_pred_test))


y_probas = svm_model.predict_proba(x_test_transformed)
fpr, tpr, _ = roc_curve(y_test, y_probas[:, 1], pos_label='Non-Fatal Injury')
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM')
plt.legend(loc="lower right")
plt.show()


# Print classification report and confusion matrix
report = classification_report(y_test, y_pred_test)
print("Classification report of SVM model before tuning")
print(report)


cfs_matrix = confusion_matrix(y_test, y_pred_test)
display = ConfusionMatrixDisplay(cfs_matrix)
display.plot()
plt.title("Confusion Matrix before tuning SVM model")
plt.show()


# Save the trained model and pipeline
dump(svm_model, '../deployment/svm_model.pkl')
dump(svm_pipeline, '../deployment/svm_pipeline.pkl')


