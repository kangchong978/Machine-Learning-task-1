

import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('test.csv') 

# print(dataset.head())
# print(dataset.describe())
# print(dataset.info())

# print(dataset.nunique())
# print(dataset.head())

# print(dataset.columns)
# print(dataset['Month'].value_counts())
# print(dataset['Occupation'].value_counts())
# print(dataset['Type_of_Loan'].value_counts())
# print(dataset['Credit_Mix'].value_counts())
# print(dataset['Payment_of_Min_Amount'].value_counts())
# print(dataset['Payment_Behaviour'].value_counts())

# dataset = dataset.dropna()

# Drop unused
dataset = dataset.drop(['Name','SSN', 'Month' ],axis=1)

min_age = 0
max_age = 120
def clean_age(age):
    try:
        # Try to convert to integer
        age_int = int(float(age))
        # Check if age is within valid range
        if min_age <= age_int <= max_age:
            return age_int
        else:
            return np.nan
    except ValueError:
        # If conversion fails, return NaN
        return np.nan

dataset['Age'] = dataset['Age'].apply(clean_age)
 
def clean_occupation(occupation):
    try:
        if occupation != '_______':
            return occupation
        else:
            return np.nan
    except ValueError:
        # If conversion fails, return NaN
        return np.nan

dataset['Occupation'] = dataset['Occupation'].apply(clean_occupation)
 
min_accounts = 0
max_accounts = 100
def clean_num_bank_accounts(accounts):
    try:
        if min_accounts <= accounts <= max_accounts:
            return accounts
        else:
            return np.nan
    except ValueError: 
        return np.nan
    
dataset['Num_Bank_Accounts'] = dataset['Num_Bank_Accounts'].apply(clean_num_bank_accounts)

min_cards = 0
max_cards = 1000
def clean_num_bank_cards(cards):
    try:
        if min_cards <= cards <= max_cards:
            return cards
        else:
            return np.nan
    except ValueError: 
        return np.nan

dataset['Num_Credit_Card'] = dataset['Num_Credit_Card'].apply(clean_num_bank_cards)
 
def clean_credit_mix(credit):
    try:
        if  credit != '_':
            return credit
        else:
            return np.nan
    except ValueError: 
        return np.nan

dataset['Credit_Mix'] = dataset['Credit_Mix'].apply(clean_credit_mix)

def clean_Outstanding_Debt(debt):
    if isinstance(debt, str):
        # Remove any leading/trailing whitespace and underscores
        debt = debt.strip().strip('_')
        
        # Check if the string is a valid integer or decimal number
        if not re.match(r'^\d+(\.\d+)?$', debt):
            return np.nan
    
    try:
        # Convert to float
        debt_float = float(debt)
        # Check if the debt is within a reasonable range (e.g., $0 to $10 million)
        if 0 <= debt_float <= 10_000_000:
            return debt_float
        else:
            return np.nan
    except ValueError:
        # If conversion fails, return NaN
        return np.nan

# Apply the cleaning function to the Annual_Income column
dataset['Outstanding_Debt'] = dataset['Outstanding_Debt'].apply(clean_Outstanding_Debt)

def clean_monthly_balance(balance):
    if isinstance(balance, str):
        # Remove any leading/trailing whitespace and underscores
        balance = balance.strip().strip('_')
        
        # Check if the string is a valid integer or decimal number
        if not re.match(r'^\d+(\.\d+)?$', balance):
            return np.nan
    
    try:
        # Convert to float
        balance_float = float(balance)
        # Check if the balance is within a reasonable range (e.g., $0 to $10 million)
        if 0 <= balance_float <= 10_000_000:
            return balance_float
        else:
            return np.nan
    except ValueError:
        # If conversion fails, return NaN
        return np.nan

# Apply the cleaning function to the Annual_Income column
dataset['Monthly_Balance'] = dataset['Monthly_Balance'].apply(clean_monthly_balance)

def clean_Credit_History_Age(age):
    if isinstance(age, str):
        match = re.match(r'(\d+)\s*Years?\s*(?:and)?\s*(\d+)?\s*Months?', age, re.IGNORECASE)
        if match:
            years = int(match.group(1))
            months = int(match.group(2)) if match.group(2) else 0
            total_months = years * 12 + months
            return total_months
    return np.nan

# Apply the cleaning function to the Credit_History_Age column
dataset['Credit_History_Age'] = dataset['Credit_History_Age'].apply(clean_Credit_History_Age)

def clean_Num_of_Delayed_Payment(value):
    if isinstance(value, str):
        # Remove any leading/trailing whitespace and underscores
        value = value.strip().strip('_')
        # Check if the string is a valid non-negative integer
        if not re.match(r'^\d+$', value):
            return np.nan
    
    try:
        # Convert to integer
        value_int = int(float(value))
        # Check if the value is non-negative and within a reasonable range
        if 0 <= value_int <= 1000:  # Assuming a reasonable maximum of 1000 delayed payments
            return value_int
        else:
            return np.nan
    except ValueError:
        # If conversion fails, return NaN
        return np.nan

# Apply the cleaning function to the Num_of_Delayed_Payment column
dataset['Num_of_Delayed_Payment'] = dataset['Num_of_Delayed_Payment'].apply(clean_Num_of_Delayed_Payment)

def clean_Num_of_Loan(value):
    if isinstance(value, str):
        # Remove any leading/trailing whitespace and underscores
        value = value.strip().strip('_')
        # Check if the string is a valid non-negative integer
        if not re.match(r'^\d+$', value):
            return np.nan
    
    try:
        # Convert to integer
        value_int = int(float(value))
        # Check if the value is within the allowed range (0 to 50)
        if 0 <= value_int <= 50:
            return value_int
        else:
            return np.nan
    except ValueError:
        # If conversion fails, return NaN
        return np.nan

# Apply the cleaning function to the Num_of_Loan column
dataset['Num_of_Loan'] = dataset['Num_of_Loan'].apply(clean_Num_of_Loan)


print(dataset.nunique())

label_encoder = LabelEncoder()

dataset['Occupation'] = label_encoder.fit_transform(dataset['Occupation'])

# List of columns that have been preprocessed
preprocessed_columns = [
    'Customer_ID',  # Assuming this is the unique identifier
    'Age',
    'Occupation',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',  # This wasn't preprocessed but was analyzed
    'Credit_Mix',
    'Outstanding_Debt',
    'Monthly_Balance',
    'Credit_History_Age',  # Add this to the list of preprocessed columns
    'Num_of_Delayed_Payment',
    'Num_of_Loan'
]

# Select only the preprocessed columns
dataset_preprocessed = dataset[preprocessed_columns]

# Define aggregation dictionary
agg_dict = {
    'Age': 'mean',
    'Occupation': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    'Num_Bank_Accounts': 'mean',
    'Num_Credit_Card': 'mean',
    'Interest_Rate': 'mean',
    'Credit_Mix': lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan, 
    'Outstanding_Debt':'mean',
    'Monthly_Balance':'mean',
    'Credit_History_Age': 'max',
    'Num_of_Delayed_Payment': 'mean' ,
    'Num_of_Loan':'mean'
}

dataset['Credit_Mix'] = label_encoder.fit_transform(dataset['Credit_Mix'])

# Perform the groupby and aggregation
merged_data = dataset_preprocessed.groupby('Customer_ID').agg(agg_dict).reset_index()

merged_data = merged_data.dropna(subset=['Credit_Mix'])

# Print info about the final merged dataset
print(f"Final merged dataset shape: {merged_data.shape}")
print("\nMerged data info:")
print(merged_data.info()) 

# Optional: Display the first few rows of the merged dataset
print("\nFirst few rows of the merged dataset:")
print(merged_data.head())

# Optional: Check for any remaining duplicates
duplicates = merged_data[merged_data.duplicated('Customer_ID')]
if not duplicates.empty:
    print(f"\nWarning: There are {len(duplicates)} duplicate Customer_IDs in the merged dataset.")
else:
    print("\nNo duplicate Customer_IDs in the merged dataset.")
    
merged_data = merged_data.drop(['Customer_ID' ],axis=1)
merged_data = merged_data.dropna()

x = merged_data.drop(columns = ['Credit_Mix'])
y = merged_data['Credit_Mix']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train_scaled, x_test_scaled, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=9)

# original spliter
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=9)

# Create and train the logistic regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(x_train_scaled, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(x_test_scaled)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get the original labels
original_labels = label_encoder.classes_

# Create and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=original_labels, yticklabels=original_labels)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': x.columns,
    'importance': abs(lr_model.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Optionally, plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Logistic Regression')
plt.show()

