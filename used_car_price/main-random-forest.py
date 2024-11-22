
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Create a pipeline with preprocessor and Random Forest Regressor
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True))])

### GridCV search best parameters

# Define the parameter grid for GridSearchCV
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}
# Create GridSearchCV object
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Make predictions on the test set using the best estimator
y_pred_log = best_estimator.predict(X_test)
y_pred_train_log = best_estimator.predict(X_train)

### 

# # Fit the model
# rf_pipeline.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred_log = rf_pipeline.predict(X_test)
# y_pred_train_log = rf_pipeline.predict(X_train)

###

# Convert predictions back to original scale
y_pred = np.expm1(y_pred_log)
y_pred_train = np.expm1(y_pred_train_log)
y_test_original = np.expm1(y_test)

# Evaluate the model
# oob_score = rf_pipeline.named_steps['regressor'].oob_score_
r2 = r2_score(y_test_original, y_pred) 

# print(f"out-of-bag score: {oob_score}")
print(f"R-squared Score: {r2}")

# Calculate RMSE for test set (log scale)
mse_test_log = mean_squared_error(y_test, y_pred_log)
rmse_test_log = np.sqrt(mse_test_log)
print(f"Root Mean Squared Error (test, log scale): {rmse_test_log}")

# Calculate RMSE for train set (log scale)
mse_train_log = mean_squared_error(y_train, y_pred_train_log)
rmse_train_log = np.sqrt(mse_train_log)
print(f"Root Mean Squared Error (train, log scale): {rmse_train_log}")

# Visualize the results by grids
plt.figure(figsize=(12, 8))
plt.scatter(X_test['year'], y_test_original, color='blue', alpha=0.5, label='Actual prices')

# Create a grid for smooth prediction line
years = np.sort(X_test['year'].unique())
predictions = []

for year in years:
    # (modify for CV)
    # pred = rf_pipeline.predict(pd.DataFrame({'year': [year], 
    #                                          'distance_travelled(kms)': [X_test['distance_travelled(kms)'].median()],
    #                                          'brand': [X_test['brand'].mode()[0]]}))
    pred = best_estimator.predict(pd.DataFrame({'year': [year], 
                                             'distance_travelled(kms)': [X_test['distance_travelled(kms)'].median()],
                                             'brand': [X_test['brand'].mode()[0]]}))
    predictions.append(np.expm1(pred)[0])

plt.plot(years, predictions, color='green', label='Predicted prices')

plt.title("Random Forest Regression Results (Log Scale)")
plt.xlabel('Year')
plt.ylabel('Price (log scale)')
plt.yscale('log')
plt.legend()

# Format y-axis labels to show exponential values
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.show()

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Car Price Prediction: Actual vs Predicted (Random Forest)')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()

# Feature importance visualization
feature_names = numeric_features + [f"brand_{brand}" for brand in best_estimator.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['brand'])]
importances = best_estimator.named_steps['regressor'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Function to predict price for user input
def predict_price(year, distance, brand):
    input_data = pd.DataFrame([[year, distance, brand]], columns=['year', 'distance_travelled(kms)', 'brand'])
    predicted_log_price = best_estimator.predict(input_data)
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