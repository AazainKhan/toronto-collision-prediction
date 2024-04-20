
# Import libraries
from joblib import dump
import scikitplot as skplt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

x_train_prepared, y_train = SMOTE(
    random_state=42).fit_resample(x_train_prepared, y_train)
x_test_transformed = full_pipeline.transform(x_test)


neural_net = MLPClassifier(random_state=42)
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (100, 50, 25)],
    'activation': ['relu', 'logistic', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [200, 300, 400, 500],
    'learning_rate': ['constant', 'adaptive']
}

neu_grid = RandomizedSearchCV(
    neural_net, param_grid, cv=10, n_jobs=-1, random_state=42)
neu_grid.fit(x_train_prepared, y_train)

neural_best_params = neu_grid.best_params_
neural_best_score = neu_grid.best_score_
print("Best Parameters:", neural_best_params)
print("Best Estimator:", neu_grid.best_estimator_)
print("Best Score:", neural_best_score)


neural_best_model = neu_grid.best_estimator_
y_pred = neural_best_model.predict(x_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)


y_probas = neural_best_model.predict_proba(x_test_transformed)
print(y_probas)
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.title('ROC Curves of Neural Network Model')
plt.show()


report = classification_report(y_test, y_pred)
print("Classification report of Neural Network Model:")
print(report)


cfs_matrix = confusion_matrix(
    y_test, y_pred, labels=neural_best_model.classes_)
display = ConfusionMatrixDisplay(
    cfs_matrix, display_labels=neural_best_model.classes_)
display.plot()
plt.title("Confusion Matrix of Neural Network Model")
plt.show()


dump(neural_best_model, '../deployment/nn_model.pkl')


