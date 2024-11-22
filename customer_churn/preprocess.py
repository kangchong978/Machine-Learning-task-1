import re
import numpy as np

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
    
def clean_occupation(occupation):
    try:
        if occupation != '_______':
            return occupation
        else:
            return np.nan
    except ValueError:
        # If conversion fails, return NaN
        return np.nan
    
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
    
def clean_credit_mix(credit):
    try:
        if  credit != '_':
            return credit
        else:
            return np.nan
    except ValueError: 
        return np.nan

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
    
def clean_Credit_History_Age(age):
    if isinstance(age, str):
        match = re.match(r'(\d+)\s*Years?\s*(?:and)?\s*(\d+)?\s*Months?', age, re.IGNORECASE)
        if match:
            years = int(match.group(1))
            months = int(match.group(2)) if match.group(2) else 0
            total_months = years * 12 + months
            return total_months
    return np.nan

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