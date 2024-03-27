# Import libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('KSI.csv')

# Print the shape of the data
print(df.shape)

print()

# Print the first 5 rows of the data
print(df.head())

print()

# Print the info of the data
df.info()

print()

# Print the summary statistics of the data
print(df.describe())

print()

# Check for missing values
print(df.isnull().sum())


# Plot histogram
df.hist(bins=50,figsize=(20,20))
plt.show()

# Plot correlation matrix of numerical columns
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='RdYlGn')
plt.show()

#* From correlation matrix, we can see that 'X' and 'LONGITUDE', 'Y' and 'LATITUDE' are highly correlated. So, we can drop one of them.
columns_to_drop = ['X', 'Y']
df = df.drop(columns=columns_to_drop)
# Get the columns which are boolean

bool_cols = [df.columns[col] for col in range(36,49)]

#Fill missing values with 'No' in boolean columns
df[bool_cols] = df[bool_cols].fillna('No')

# Drop columns having more than 80% missing values
missing_percentages = (df.isnull().sum() / len(df)) * 100
columns_to_drop = missing_percentages[missing_percentages > 80].index
df = df.drop(columns=columns_to_drop)

# Drop columns which may not be useful for analysis
columns_to_drop = ['ObjectId', 'INDEX_', 'ACCNUM', 'STREET1', 'STREET2', 'DISTRICT', 'WARDNUM', 'DIVISION', 'NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140']
'''
Explanation:
ObjectId, INDEX_, ACCNUM: These columns are unique for each row
NEIGHBOURHOOD_158, NEIGHBOURHOOD_140: These columns are indentical to 'HOOD_158' and 'HOOD_140' respectively
STREET1, STREET2: According to the requirement, model will predict the severity of accident in certain neighbourhoods
'''

#! 'INJURY' feature which might be dropped depends on the accuracy score of model
df = df.drop(columns=columns_to_drop)

# Convert 'property' to 'non fatal'
df['ACCLASS'] = df['ACCLASS'].str.replace("Property Damage Only","Non-Fatal Injury")

# Categorical columns which have <3% missing values, we can drop them. The number is not remarkable, it won't affect the accuracy, it is just below 3%
cat_cols =  df.select_dtypes(include='object') # get only categorical columns
missing_percentages = cat_cols.isnull().sum()/len(df) * 100
cat_col_val_drop = missing_percentages[missing_percentages <= 3].index
cat_col_val_drop
df = df.dropna(subset=cat_col_val_drop)

print()
print(df.isnull().sum())

# Fill null values to naN
df = df.fillna(value=np.nan)

# Convert 'TIME' to 'AM_PM', we can use this feature to check the percentage of accidents happened in day and night.
#! Note: This feature is only used for data exploration, it won't be used for training the model.
df['AM_PM'] = df['TIME'].apply(lambda x: 'AM' if x < 1200 else 'PM')

# Extract 'DATE' to 'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK'
#! Note: These features are only used for data exploration, it won't be used for training the model.
df['DATE'] = pd.to_datetime(df['DATE']) # Convert 'DATE' to correct datetime format before extracting
df['MONTH'] = df['DATE'].dt.month_name() # Get months
df['DAY'] = df['DATE'].dt.day # Get days
df['DAY_OF_WEEK'] = df['DATE'].dt.day_name() #Get days of week
df = df.drop(columns=['DATE'], axis=1) # Drop 'DATE' column

# Plot the number of accidents happened in day and night
plt.figure(figsize=(25,20))
plt.bar(df['AM_PM'].unique(), df['AM_PM'].value_counts().values)
plt.title('Number of accidents happened in day and night')
plt.xlabel('Day/Night')
plt.ylabel('Number of accidents')
plt.show()


# Plot the number of accidents happened in each month
plt.figure(figsize=(25,20))
df['MONTH'].value_counts().reindex(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']).plot(kind='bar', figsize=(10, 6)).plot(kind='bar', figsize=(10, 6))
plt.title('Number of Accidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()


# Plot the number of accidents happened in day of week
plt.figure(figsize=(25,20))
# Convert day of week from index to string
df['DAY_OF_WEEK'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Number of accidents happened in day of week')
plt.ylabel('')
plt.show()


# Plot the number of accidents by INVAGE
plt.figure(figsize=(25,20))
plt.bar(df['INVAGE'].unique(), df['INVAGE'].value_counts().values)
plt.title('Number of accidents by INVAGE')
plt.xlabel('INVAGE')
plt.ylabel('Number of accidents')
plt.show()

# Scatter plot of LATITUDE vs LONGITUDE
fatal_accidents = df[df['ACCLASS'] == 'Fatal']
non_fatal_accidents = df[df['ACCLASS'] != 'Fatal']
plt.figure(figsize=(10, 6))
# Plot non-fatal accidents
plt.scatter(non_fatal_accidents['LONGITUDE'], non_fatal_accidents['LATITUDE'], alpha=0.5, label='Non-Fatal', color='blue')
# Plot fatal accidents
plt.scatter(fatal_accidents['LONGITUDE'], fatal_accidents['LATITUDE'], alpha=0.5, label='Fatal', color='red')
plt.title('Spatial Distribution of Traffic Accidents')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

# Plot Accident Severity by Impact Type
df.groupby('IMPACTYPE')['ACCLASS'].value_counts().unstack().plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Accident Severity by Impact Type')
plt.xlabel('Impact Type')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

# Using chi-square test to find correlation between categorical columns and 'ACCLASS' column
cat_df = df.select_dtypes(include='object')
ordinal_encoder = OrdinalEncoder()
cat_df = ordinal_encoder.fit_transform(cat_df)
cat_df = pd.DataFrame(cat_df, columns=df.select_dtypes(include='object').columns)


X = cat_df.drop(columns=['ACCLASS'], axis=1)
feature_names = list(X.columns)
X = SimpleImputer(strategy='most_frequent').fit_transform(X)
y = cat_df['ACCLASS']
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(feature_names, columns=['Specs'])
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']

# Ascending order
featureScores = featureScores.sort_values(by='Score', ascending=False)


# Plot the result
plt.figure(figsize=(40,30))
sns.barplot(x='Specs', y='Score', data=featureScores)
plt.title('Chi-square test result')
plt.xlabel('Features')
plt.ylabel('Score')
plt.show()



# Create new variable to use RFE
# Using model to find important features of the original dataset
df2 = pd.read_csv('KSI.csv')
df2['ACCLASS'] = df2['ACCLASS'].str.replace("Property Damage Only", "Non-Fatal Injury")
df2 = df2.fillna(value=np.nan)


df2 = df2.dropna(subset=['ACCLASS'])
df2 = df2.drop(['ObjectId', 'X', 'Y','NEIGHBOURHOOD_140', 'NEIGHBOURHOOD_158'], axis = 1)

X = df2.drop(columns=['ACCLASS'], axis=1)
y = df2['ACCLASS']

num_features = X.select_dtypes(include='number').columns
cat_features = X.select_dtypes(include='object').columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
])


X_prepared = full_pipeline.fit_transform(X)
X_prepared = pd.DataFrame(X_prepared, columns=num_features.tolist() + cat_features.tolist())
X_prepared = X_prepared.apply(LabelEncoder().fit_transform)
# Try using Ordinal Encoder

estimator = LogisticRegression(random_state=5)
selector = RFE(estimator, step=1, n_features_to_select=10)
selector = selector.fit(X_prepared, y)
ranking = selector.ranking_
ranking = pd.DataFrame(ranking, index=X_prepared.columns, columns=['Rank'])
ranking = ranking.sort_values(by='Rank', ascending=True)

# Plot the result
plt.figure(figsize=(40,30))
sns.barplot(x=ranking['Rank'], y=ranking.index)
plt.title('RFE result')
plt.xlabel('Rank')
plt.ylabel('Features')
plt.show()

# Check unique values of 'HOOD_158' and 'HOOD_140' columns
print(df['HOOD_158'].unique())

#* We can see that 'NSA' is present in 'HOOD_158' and 'HOOD_140' columns. This value doesn't make sense, so we can drop it.
cols = ['HOOD_158', 'HOOD_140']
df[cols] = df[cols].replace("NSA", np.nan)
df = df.dropna(subset=['HOOD_158', 'HOOD_140'])

columns_to_drop = ['INJURY','YEAR', 'TIME','ROAD_CLASS', 'LOCCOORD' ,'ACCLOC', 'TRAFFCTL', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND', 'INVAGE', 'TRUCK',  'MONTH', 'DAY','AM_PM', 'DAY_OF_WEEK','PASSENGER', 'ALCOHOL', 'DISABILITY', 'INITDIR', 'LONGITUDE', 'LATITUDE']
# Drop 'INJURY' because its coefficient is low
# We also drop 'LONGITUDE' and 'LATITUDE' because according to the requirement, model will predict the severity of accident in certain neighbourhoods
df_official = df.drop(columns=columns_to_drop, axis=1)
df_official[cols] = df_official[cols].astype(np.number)

# Save cleaned data with SMOTE
df_official.to_csv('cleaned_data_KSI.csv', index=False)

# Import cleaned file
df_official = pd.read_csv('cleaned_data_KSI.csv')

# Print info of the data
df_official.info()

x_group3 = df_official.drop(columns=['ACCLASS'], axis=1)
y_group3 = df_official['ACCLASS']


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=5)
for train_index, test_index in sss.split(x_group3, y_group3):
    x_train, x_test = x_group3.iloc[train_index], x_group3.iloc[test_index]
    y_train, y_test = y_group3.iloc[train_index], y_group3.iloc[test_index]


print("Before using SMOTE")
print(y_train.value_counts())
print()


x_train, y_train = SMOTENC(random_state = 5, categorical_features=[0,1,2,3,4,5,6,7,8, 9, 10]).fit_resample(x_train, y_train)

#! Important note: SMOTE cannot handle more than 15 features
print("After using SMOTE")
print(y_train.value_counts())
print()



num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer([
    ('num', num_pipeline, x_train.select_dtypes(include='number').columns),
    ('cat', cat_pipeline, x_train.select_dtypes(include='object').columns)
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

# Test the pipeline
x_train_prepared = full_pipeline.fit_transform(x_train)


log_reg = LogisticRegression(random_state=5)
log_reg.fit(x_train_prepared, y_train)

y_pred = log_reg.predict(x_train_prepared)
print("Accuracy score on training data: ", accuracy_score(y_train, y_pred))

x_test_transformed = full_pipeline.transform(x_test)
y_pred = log_reg.predict(x_test_transformed)
print("Accuracy score on testing data: ", accuracy_score(y_test, y_pred))
