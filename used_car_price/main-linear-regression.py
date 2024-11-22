import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Dataset.csv')

# Apply logarithmic transformation to price
df['log_price'] = np.log1p(df['price'])

# Prepare the features and target
X = df[['year', 'distance_travelled(kms)', 'brand']]
y = df['log_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing steps for numerical and categorical features
numeric_features = ['year', 'distance_travelled(kms)']
categorical_features = ['brand']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessor and Linear Regression model
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', LinearRegression())])

# Fit the model
lr_pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred_log = lr_pipeline.predict(X_test)
y_pred_train_log = lr_pipeline.predict(X_train)

# Convert predictions back to original scale
y_pred = np.expm1(y_pred_log)
y_pred_train = np.expm1(y_pred_train_log)
y_test_original = np.expm1(y_test)


# Evaluate the model
r2 = r2_score(y_test_original, y_pred)
print(f"R-squared Score: {r2}")

# Calculate RMSE for test set (original scale)
# mse_test_original = mean_squared_error(y_test_original, y_pred)
# rmse_test_original = np.sqrt(mse_test_original)
# print(f"Root Mean Squared Error (test, original scale): {rmse_test_original}")

# Calculate RMSE for test set (log scale)
mse_test_log = mean_squared_error(y_test, y_pred_log)
rmse_test_log = np.sqrt(mse_test_log)
print(f"Root Mean Squared Error (test, log scale): {rmse_test_log}")

# Calculate RMSE for train set (log scale)
mse_train_log = mean_squared_error(y_train, y_pred_train_log)
rmse_train_log = np.sqrt(mse_train_log)
print(f"Root Mean Squared Error (train, log scale): {rmse_train_log}")
plt.figure(figsize=(12, 8))

# Scatter plot of actual prices
plt.scatter(X_test['year'], y_test_original, alpha=0.5, color='blue', label='Actual Prices')

# Generate predictions for a range of years
years_range = np.linspace(X_test['year'].min(), X_test['year'].max(), 100).reshape(-1, 1)
X_pred = pd.DataFrame({
    'year': years_range.ravel(),
    'distance_travelled(kms)': [X_test['distance_travelled(kms)'].median()] * 100,
    'brand': [X_test['brand'].mode()[0]] * 100
})
y_pred_range = np.expm1(lr_pipeline.predict(X_pred))

# Plot the prediction line
plt.plot(years_range, y_pred_range, color='red', linewidth=2, label='Predicted Prices')

plt.title('Car Price Prediction: Year vs Price (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Price (log scale)')
plt.yscale('log')
plt.legend()

# Format y-axis labels to show exponential values
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.show()

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Car Price Prediction: Actual vs Predicted (Linear Regression)')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()

# Feature importance visualization (coefficients for linear regression)
feature_names = numeric_features + [f"brand_{brand}" for brand in lr_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['brand'])]
coefficients = lr_pipeline.named_steps['regressor'].coef_
importances = np.abs(coefficients)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importances (Absolute Coefficients)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Function to predict price for user input
def predict_price(year, distance, brand):
    input_data = pd.DataFrame([[year, distance, brand]], columns=['year', 'distance_travelled(kms)', 'brand'])
    predicted_log_price = lr_pipeline.predict(input_data)
    predicted_price = np.expm1(predicted_log_price)[0]
    return predicted_price

# Example usage of the prediction function
user_year = 2017
user_distance = 64593
user_brand = 'Toyota'  # Replace with an actual brand from your dataset
predicted_price = predict_price(user_year, user_distance, user_brand)
print(f"\nPredicted price for a {user_year} {user_brand} with {user_distance} km traveled: ${predicted_price:.2f}")

# Print top 10 most important features
top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 most important features:")
for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")