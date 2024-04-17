# Import libraries
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
import scikitplot as skplt
from joblib import dump


df_official = pd.read_csv('cleaned_data_KSI.csv')
df_official.info()


print()
print(df_official.describe().T)
x_group3 = df_official.drop(columns=['ACCLASS'], axis=1)
y_group3 = df_official['ACCLASS']


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(x_group3, y_group3):
    x_train, x_test = x_group3.loc[train_index], x_group3.loc[test_index]
    y_train, y_test = y_group3.loc[train_index], y_group3.loc[test_index]


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

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

x_train_prepared = full_pipeline.fit_transform(x_train)

x_train_prepared, y_train = SMOTE(random_state=42).fit_resample(x_train_prepared, y_train)
x_test_transformed = full_pipeline.transform(x_test)

log_reg = LogisticRegression(random_state=42)
param_grid = {
    'C': np.logspace(-4, 4, 50),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300, 400, 500]
}

grid = RandomizedSearchCV(log_reg, param_grid, cv=5, n_jobs=-1, scoring='accuracy', n_iter=100, random_state=5)
grid.fit(x_train_prepared, y_train)

print("Best parameters: ", grid.best_params_)
print("Best cross-validation score: ", grid.best_score_)

log_reg_best = grid.best_estimator_
y_pred = log_reg_best.predict(x_test_transformed)
print("Accuracy score on testing data with best parameters: ", accuracy_score(y_test, y_pred))

y_probas = log_reg_best.predict_proba(x_test_transformed)
print(y_probas)
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.title('ROC Curves of Logistic Regression model')
plt.show()

report = classification_report(y_test, y_pred)
print("Classification report of Logistic Regression")
print(report)

cfs_matrix = confusion_matrix(y_test, y_pred, labels=log_reg_best.classes_)
display = ConfusionMatrixDisplay(cfs_matrix,display_labels=log_reg_best.classes_)
display.plot()
plt.title("Confusion Matrix after tuning model")
plt.show()

dump(log_reg_best, './deployment/log_reg_model.pkl')
dump(full_pipeline, './deployment/pipeline.pkl')

