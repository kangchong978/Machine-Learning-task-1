import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
dataset = pd.read_csv('test.csv')

# Display basic information about the dataset
print(dataset.head())
print(dataset.describe())
print(dataset.info())
print(dataset.nunique())
print(dataset.head())
print(dataset.columns)

# Check the unique values and their counts for the 'occupation' column
occupation_counts = dataset['Occupation'].value_counts()

# Plot the distribution of the 'occupation' column
plt.figure(figsize=(10, 6))
occupation_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Occupations')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
plt.tight_layout()
plt.show()

# Basic statistics for the column
print("Monthly_Inhand_Salary Statistics:")
print(dataset['Monthly_Inhand_Salary'].describe())

# Plot the distribution of Monthly_Inhand_Salary
plt.figure(figsize=(10, 6))
plt.hist(dataset['Monthly_Inhand_Salary'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Monthly Inhand Salary')
plt.xlabel('Monthly Inhand Salary')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

credit_mix_counts = dataset['Credit_Mix'].value_counts()

plt.figure(figsize=(10, 6))
credit_mix_counts.plot(kind='bar', color='salmon')
plt.title('Distribution of Credit Mix')
plt.xlabel('Credit Mix')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
 

# Convert 'Age' to numeric, invalid entries will be NaN
dataset['Age'] = pd.to_numeric(dataset['Age'], errors='coerce')

# Define bins and labels
bins = [-float('inf'), 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
labels = ['<1', '1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '>100']

# Create age groups, treating NaN values as 'Invalid'
dataset['Age Group'] = pd.cut(dataset['Age'], bins=bins, labels=labels)
dataset['Age Group'] = dataset['Age Group'].cat.add_categories(['Invalid'])
dataset['Age Group'].fillna('Invalid', inplace=True)

# Count the frequency of each age group
age_group_counts = dataset['Age Group'].value_counts().sort_index()

# Plot the distribution of age groups
plt.figure(figsize=(10, 6))
age_group_counts.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Age Group Distribution (with Invalid Data)')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()