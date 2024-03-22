# Import libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer

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
bool_cols = [df.columns[col] for col in range(38,51)]

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