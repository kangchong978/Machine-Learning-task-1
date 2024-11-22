# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# # Load the data
# df = pd.read_csv('Dataset.csv')

# # Preprocess the data
# X = df[['year']]  # We're only using 'year' as the feature
# y = df['price']

# # Apply feature scaling to the price
# scaler = MinMaxScaler(feature_range=(0, 0.4))
# y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# # Create and train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"Root Mean Squared Error: {rmse}")
# print(f"R-squared Score: {r2}")

# # Print the model coefficients
# print(f"Intercept: {model.intercept_}")
# print(f"Coefficient: {model.coef_[0]}")

# # Visualize the results
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Prices')
# plt.plot(X_test, y_pred, color='red', label='Predicted Prices')
# plt.xlabel('Year')
# plt.ylabel('Price')
# plt.title('Car Price Prediction: Actual vs Predicted')
# plt.legend()
# plt.show()

# # Predict prices for specific years
# years_to_predict = [2020, 2021, 2022, 2023, 2024]
# predicted_prices = model.predict([[year] for year in years_to_predict])

# print("\nPredicted prices for specific years:")
# for year, price in zip(years_to_predict, predicted_prices):
#     print(f"Year {year}: ${price:.2f}")


# 2nd 

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # Load the data
# df = pd.read_csv('Dataset.csv')

# # Prepare the features and target
# X = df[['year', 'distance_travelled(kms)']]
# y = df['price']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Create and train the linear regression model
# model = LinearRegression()
# model.fit(X_train_scaled, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test_scaled)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"Root Mean Squared Error: {rmse}")
# print(f"R-squared Score: {r2}")

# # Print the model coefficients
# feature_names = ['year', 'distance_travelled(kms)']
# for name, coef in zip(feature_names, model.coef_):
#     print(f"Coefficient for {name}: {coef}")
# print(f"Intercept: {model.intercept_}")

# # Visualize the results
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Car Price Prediction: Actual vs Predicted')
# plt.tight_layout()
# plt.show()

# # Feature importance visualization
# importance = np.abs(model.coef_)
# plt.figure(figsize=(10, 6))
# plt.bar(feature_names, importance)
# plt.title('Feature Importance in Car Price Prediction')
# plt.xlabel('Features')
# plt.ylabel('Absolute Coefficient Value')
# plt.tight_layout()
# plt.show()

# # Predict prices for specific scenarios
# print("\nPredictions for specific scenarios:")
# scenarios = [
#     {'year': 2020, 'distance_travelled(kms)': 50000},
#     {'year': 2018, 'distance_travelled(kms)': 100000},
#     {'year': 2015, 'distance_travelled(kms)': 150000}
# ]

# for scenario in scenarios:
#     scenario_scaled = scaler.transform(pd.DataFrame([scenario]))
#     predicted_price = model.predict(scenario_scaled)[0]
#     print(f"Year: {scenario['year']}, Distance: {scenario['distance_travelled(kms)']} km")
#     print(f"Predicted Price: ${predicted_price:.2f}\n")

# # Function to predict price for user input
# def predict_price(year, distance):
#     input_data = pd.DataFrame([[year, distance]], columns=['year', 'distance_travelled(kms)'])
#     input_scaled = scaler.transform(input_data)
#     predicted_price = model.predict(input_scaled)[0]
#     return predicted_price

# # Example usage of the prediction function
# user_year = 2019
# user_distance = 75000
# predicted_price = predict_price(user_year, user_distance)
# print(f"\nPredicted price for a {user_year} car with {user_distance} km traveled: ${predicted_price:.2f}")

# 3rd

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # Load the data
# df = pd.read_csv('Dataset.csv')

# # Apply logarithmic transformation to price
# df['log_price'] = np.log1p(df['price'])

# # Prepare the features and target
# X = df[['year', 'distance_travelled(kms)']]
# y = df['log_price']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# X_scaler = StandardScaler()
# X_train_scaled = X_scaler.fit_transform(X_train)
# X_test_scaled = X_scaler.transform(X_test)

# # Create and train the linear regression model
# model = LinearRegression()
# model.fit(X_train_scaled, y_train)

# # Make predictions on the test set
# y_pred_log = model.predict(X_test_scaled)

# # Convert predictions back to original scale
# y_pred = np.expm1(y_pred_log)
# y_test_original = np.expm1(y_test)

# # Evaluate the model
# mse = mean_squared_error(y_test_original, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test_original, y_pred)

# print(f"Root Mean Squared Error: {rmse}")
# print(f"R-squared Score: {r2}")

# # Print the model coefficients
# feature_names = ['year', 'distance_travelled(kms)']
# for name, coef in zip(feature_names, model.coef_):
#     print(f"Coefficient for {name}: {coef}")
# print(f"Intercept: {model.intercept_}")

# # Visualize the results
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test_original, y_pred, alpha=0.5)
# plt.plot([y_test_original.min(), y_test_original.max()], 
#          [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Car Price Prediction: Actual vs Predicted (Log-Scaled)')
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.show()

# # Feature importance visualization
# importance = np.abs(model.coef_)
# plt.figure(figsize=(10, 6))
# plt.bar(feature_names, importance)
# plt.title('Feature Importance in Car Price Prediction')
# plt.xlabel('Features')
# plt.ylabel('Absolute Coefficient Value')
# plt.tight_layout()
# plt.show()

# # Function to predict price for user input
# def predict_price(year, distance):
#     input_data = pd.DataFrame([[year, distance]], columns=['year', 'distance_travelled(kms)'])
#     input_scaled = X_scaler.transform(input_data)
#     predicted_log_price = model.predict(input_scaled)
#     predicted_price = np.expm1(predicted_log_price)[0]
#     return predicted_price

# # Example usage of the prediction function
# user_year = 2019
# user_distance = 75000
# predicted_price = predict_price(user_year, user_distance)
# print(f"\nPredicted price for a {user_year} car with {user_distance} km traveled: ${predicted_price:.2f}")

# 4 th

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # Load the data
# df = pd.read_csv('Dataset.csv')

# # Apply logarithmic transformation to price
# df['log_price'] = np.log1p(df['price'])

# # Prepare the features and target
# X = df[['year', 'distance_travelled(kms)']]
# y = df['log_price']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Create and train the Random Forest model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train_scaled, y_train)

# # Make predictions on the test set
# y_pred_log = model.predict(X_test_scaled)

# # Convert predictions back to original scale
# y_pred = np.expm1(y_pred_log)
# y_test_original = np.expm1(y_test)

# # Evaluate the model
# mse = mean_squared_error(y_test_original, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test_original, y_pred)

# print(f"Root Mean Squared Error: {rmse}")
# print(f"R-squared Score: {r2}")

# # Visualize the results
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test_original, y_pred, alpha=0.5)
# plt.plot([y_test_original.min(), y_test_original.max()], 
#          [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Car Price Prediction: Actual vs Predicted (Random Forest)')
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.show()

# # Feature importance visualization
# importance = model.feature_importances_
# feature_names = ['year', 'distance_travelled(kms)']
# plt.figure(figsize=(10, 6))
# plt.bar(feature_names, importance)
# plt.title('Feature Importance in Car Price Prediction')
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.tight_layout()
# plt.show()

# # Function to predict price for user input
# def predict_price(year, distance):
#     input_data = pd.DataFrame([[year, distance]], columns=['year', 'distance_travelled(kms)'])
#     input_scaled = scaler.transform(input_data)
#     predicted_log_price = model.predict(input_scaled)
#     predicted_price = np.expm1(predicted_log_price)[0]
#     return predicted_price

# # Example usage of the prediction function
# user_year = 2017
# user_distance = 64593
# predicted_price = predict_price(user_year, user_distance)
# print(f"\nPredicted price for a {user_year} car with {user_distance} km traveled: ${predicted_price:.2f}")

# 5th

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # Load the data
# df = pd.read_csv('Dataset.csv')

# # Apply logarithmic transformation to price
# df['log_price'] = np.log1p(df['price'])

# # Prepare the features and target
# X = df[['year', 'distance_travelled(kms)', 'brand']]
# y = df['log_price']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create preprocessing steps for numerical and categorical features
# numeric_features = ['year', 'distance_travelled(kms)']
# categorical_features = ['brand']

# numeric_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # Create a pipeline with preprocessor and Random Forest model
# model = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# # Fit the model
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred_log = model.predict(X_test)

# # Convert predictions back to original scale
# y_pred = np.expm1(y_pred_log)
# y_test_original = np.expm1(y_test)

# # Evaluate the model
# mse = mean_squared_error(y_test_original, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test_original, y_pred)

# print(f"Root Mean Squared Error: {rmse}")
# print(f"R-squared Score: {r2}")

# # Visualize the results
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test_original, y_pred, alpha=0.5)
# plt.plot([y_test_original.min(), y_test_original.max()], 
#          [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
# plt.xlabel('Actual Price')
# plt.ylabel('Predicted Price')
# plt.title('Car Price Prediction: Actual vs Predicted (Random Forest with Brand)')
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.show()

# # Feature importance visualization
# feature_names = numeric_features + [f"brand_{brand}" for brand in model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['brand'])]
# importances = model.named_steps['regressor'].feature_importances_
# indices = np.argsort(importances)[::-1]

# plt.figure(figsize=(12, 8))
# plt.title("Feature Importances")
# plt.bar(range(len(importances)), importances[indices])
# plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
# plt.tight_layout()
# plt.show()

# # Function to predict price for user input
# def predict_price(year, distance, brand):
#     input_data = pd.DataFrame([[year, distance, brand]], columns=['year', 'distance_travelled(kms)', 'brand'])
#     predicted_log_price = model.predict(input_data)
#     predicted_price = np.expm1(predicted_log_price)[0]
#     return predicted_price

# # Example usage of the prediction function
# user_year = 2017
# user_distance = 64593
# user_brand = 'Toyota'  # Replace with an actual brand from your dataset
# predicted_price = predict_price(user_year, user_distance, user_brand)
# print(f"\nPredicted price for a {user_year} {user_brand} with {user_distance} km traveled: ${predicted_price:.2f}")

# # Print top 10 most important features
# top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
# print("\nTop 10 most important features:")
# for feature, importance in top_features:
#     print(f"{feature}: {importance:.4f}")

# 6th

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# Create a pipeline with preprocessor and Random Forest model
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(random_state=42))])

# Define the parameter grid for GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Create GridSearchCV object
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred_log = best_model.predict(X_test)

# Convert predictions back to original scale
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# Evaluate the model
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Car Price Prediction: Actual vs Predicted (Tuned Random Forest)')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()

# Feature importance visualization
feature_names = numeric_features + [f"brand_{brand}" for brand in best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['brand'])]
importances = best_model.named_steps['regressor'].feature_importances_
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
    predicted_log_price = best_model.predict(input_data)
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