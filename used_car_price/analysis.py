import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('Dataset.csv')

# Display basic information about the dataset
print(df.info())

# Show the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Analyze the distribution of the target variable (price)
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of Car Prices')
plt.xlabel('Price')
plt.show()

# Analyze the relationship between year and price
plt.figure(figsize=(10, 6))
plt.scatter(df['year'], df['price'])
plt.title('Car Price vs. Year')
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()

# Analyze the relationship between distance travelled and price
plt.figure(figsize=(10, 6))
plt.scatter(df['distance_travelled(kms)'], df['price'])
plt.title('Car Price vs. Distance Travelled')
plt.xlabel('Distance Travelled (kms)')
plt.ylabel('Price')
plt.show()

# Analyze the distribution of prices by brand
plt.figure(figsize=(12, 6))
sns.boxplot(x='brand', y='price', data=df)
plt.title('Price Distribution by Brand')
plt.xticks(rotation=90)
plt.show()

# Count of cars by brand
brand_counts = df['brand'].value_counts()
plt.figure(figsize=(12, 6))
brand_counts.plot(kind='bar')
plt.title('Count of Cars by Brand')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

print("\nTop 10 brands by average price:")
print(df.groupby('brand')['price'].mean().sort_values(ascending=False).head(10))

print("\nBottom 10 brands by average price:")
print(df.groupby('brand')['price'].mean().sort_values(ascending=True).head(10))