from joblib import dump
import scikitplot as skplt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Import libraries
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


df_group5 = pd.read_csv('../cleaned_data_KSI.csv')
df_group5.info()

print(df_group5.describe().T)


x_group5 = df_group5.drop(columns=['ACCLASS'], axis=1)
y_group5 = df_group5['ACCLASS']


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(x_group5, y_group5):
    x_train, x_test = x_group5.loc[train_index], x_group5.loc[test_index]
    y_train, y_test = y_group5.loc[train_index], y_group5.loc[test_index]


print("Before using SMOTE")
print(y_train.value_counts())


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

dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

x_train_prepared = dt_pipeline.fit_transform(x_train)

x_train_prepared, y_train = SMOTE(
    random_state=42).fit_resample(x_train_prepared, y_train)


# Print class distribution after SMOTE
print("After using SMOTE")
print(y_train.value_counts())
print()



dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train_prepared, y_train)

from sklearn.model_selection import KFold, cross_val_score

crossvalscore = KFold(n_splits=10, random_state=42, shuffle=True)
scores = cross_val_score(dt, x_train_prepared, y_train,
                         cv=crossvalscore, scoring='accuracy')
print("Cross validation scores for 10 folds: ", scores)
print("Mean cross validation score: ", scores.mean())


y_pred = dt.predict(x_train_prepared)
print("Accuracy score on training data: ", accuracy_score(y_train, y_pred))


x_test_transformed = dt_pipeline.transform(x_test)
y_pred = dt.predict(x_test_transformed)
print("Accuracy score on testing data: ", accuracy_score(y_test, y_pred))


y_probas = dt.predict_proba(x_test_transformed)
print(y_probas)
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.title('ROC Curves of Decision Tree model')
plt.show()


report = classification_report(y_test, y_pred)
print("Classification report of Decision Tree before tuning model")
print(report)

cfs_matrix = confusion_matrix(y_test, y_pred, labels=dt.classes_)
display = ConfusionMatrixDisplay(cfs_matrix, display_labels=dt.classes_)
display.plot()
plt.title("Confusion Matrix before tuning model")
plt.show()

param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid = GridSearchCV(dt, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid.fit(x_train_prepared, y_train)

print("Best parameters: ", grid.best_params_)
print("Best cross-validation score: ", grid.best_score_)
print("Best estimator: ", grid.best_estimator_)


dt_best = grid.best_estimator_
dt_best.fit(x_train_prepared, y_train)
y_pred = dt_best.predict(x_test_transformed)
print("Accuracy score on testing data with best parameters: ",
      accuracy_score(y_test, y_pred))


report = classification_report(y_test, y_pred)
print("Classification report of Decision Tree after tuning model")
print(report)


cfs_matrix = confusion_matrix(y_test, y_pred, labels=dt.classes_)
display = ConfusionMatrixDisplay(cfs_matrix, display_labels=dt.classes_)
display.plot()
plt.title("Confusion Matrix after tuning model")
plt.show()


dump(dt_best, '../deployment/dt_model.pkl')


