import pandas as pd
import numpy as np

# Load the dataset
file_path = r"C:\Users\YounghunMun\Desktop\SupervisedLearning_274\project\KSI.csv"
data = pd.read_csv(file_path)

pd.set_option('display.max_columns', None)

print('\n\n\n############ Data Info ####################\n')
print(data.info())
print('\n\n\n############ Data sample ####################\n')
print(data.head())
print('\n\n\n############ Data Statistical Describe ####################\n')

print(data.describe().head())
#print(data.dtype())


numeric_data = data.select_dtypes(include=[np.number])
numeric_ranges = numeric_data.agg(['min', 'max'])

print(numeric_ranges)

import matplotlib.pyplot as plt

# Number of graphs per line
graphs_per_line = 3


num_columns = len(numeric_data.columns)
num_rows = -(-num_columns // graphs_per_line)  

#plt.figure(figsize=(20, 20))
plt.figure(figsize=(20, 4 * num_rows))

# Plot each column in its subplot
for i, column in enumerate(numeric_data.columns):
    plt.subplot(num_rows, graphs_per_line, i + 1)
    numeric_data[column].plot(kind='hist', bins=50, alpha=0.7)
    plt.title(f'Bar Graph of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True)

plt.tight_layout()
plt.show()
    

for col in data.select_dtypes(include=[np.number]).columns:
    data[col] = data[col].fillna(data[col].mean())

# For categorical columns, fill in missing values with the mode
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# 2. Dropping rows with any missing values
data = data.dropna()

# 3. Dropping columns with more than a certain percentage of missing values
threshold = 0.5  # Example threshold
data = data.dropna(axis='columns', thresh=int(threshold * len(data)))

# Display final info about data to show changes
print("\nFinal Data Info After Handling Missing Values:")
print(data.info())


# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Explore the unique values and their counts
for column in categorical_columns:
    print(f"\nColumn: {column}")
    print(data[column].value_counts())
    
# Fill missing values with the most frequent category (mode)
for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

data_encoded = pd.get_dummies(data, columns=categorical_columns)

selected_columns = ['DISTRICT', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'INVAGE', 'INJURY']

graphs_per_line = 3  

num_columns = len(selected_columns)
num_rows = -(-num_columns // graphs_per_line)  

# Set the figure size to be larger
plt.figure(figsize=(20, 5 * num_rows))

# Loop through each selected column to plot
for i, column in enumerate(selected_columns):
    plt.subplot(num_rows, graphs_per_line, i + 1)
    
    # Optional: Grouping rare categories
    counts = data[column].value_counts()
    threshold = 10  
    rare_categories = counts[counts < threshold].index
    data[column] = data[column].replace(rare_categories, 'Other')

    data[column].value_counts().plot(kind='bar')
    plt.title(f'Frequency of Categories in {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

import seaborn as sns

# Box Plot to show the distribution of a numeric variable across the categories (if applicable)
# Choose a numeric column as needed, here just an example
if 'Year' in data.columns:  # Replace 'Year' with a relevant numeric column from your dataset
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=column, y='Year', data=data)  # Replace 'Year' with your numeric column
    plt.title(f'Distribution of Year across {column}')
    plt.xlabel(column)
    plt.ylabel('Year')  # Replace 'Year' with your numeric column
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()