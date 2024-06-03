import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Read the JSON file
with open('ellipse_results.json', 'r') as file:
    data = json.load(file)

# Step 2: Extract the relevant features and target variables
df = pd.DataFrame(data)

# For Major Axis Length Model
X_major = df[['major_axis_length', 'mean_depth']]
y_major = df['real_major_axis_length']

# For Minor Axis Length Model
X_minor = df[['minor_axis_length', 'mean_depth']]
y_minor = df['real_minor_axis_length']

# Step 3: Split the data into training and testing sets
X_major_train, X_major_test, y_major_train, y_major_test = train_test_split(
    X_major, y_major, test_size=0.2, random_state=42)
X_minor_train, X_minor_test, y_minor_train, y_minor_test = train_test_split(
    X_minor, y_minor, test_size=0.2, random_state=42)

# Step 4: Train the regression models
model_major = LinearRegression()
model_minor = LinearRegression()

model_major.fit(X_major_train, y_major_train)
model_minor.fit(X_minor_train, y_minor_train)

# Step 5: Evaluate the models
y_major_pred = model_major.predict(X_major_test)
y_minor_pred = model_minor.predict(X_minor_test)

print(f'Major Axis Length Model:')
print(f'Mean squared error: {mean_squared_error(y_major_test, y_major_pred)}')
print(f'Coefficient of determination: {r2_score(y_major_test, y_major_pred)}')
print(f'Coefficients: {model_major.coef_}')
print(f'Intercept: {model_major.intercept_}')

print(f'Minor Axis Length Model:')
print(f'Mean squared error: {mean_squared_error(y_minor_test, y_minor_pred)}')
print(f'Coefficient of determination: {r2_score(y_minor_test, y_minor_pred)}')
print(f'Coefficients: {model_minor.coef_}')
print(f'Intercept: {model_minor.intercept_}')

# Show some test data
test_data_major = X_major_test.copy()
test_data_major['real_major_axis_length'] = y_major_test.values
test_data_major['pred_major_axis_length'] = y_major_pred

test_data_minor = X_minor_test.copy()
test_data_minor['real_minor_axis_length'] = y_minor_test.values
test_data_minor['pred_minor_axis_length'] = y_minor_pred

print("\nSample Test Data for Major Axis Length Model:")
print(test_data_major)

print("\nSample Test Data for Minor Axis Length Model:")
print(test_data_minor)

# Step 6: Predict the values using the trained models
def predict_real_major_length(major_axis_length, mean_depth):
    input_data = pd.DataFrame([[major_axis_length, mean_depth]], columns=['major_axis_length', 'mean_depth'])
    real_major_axis_length = model_major.predict(input_data)[0]
    return real_major_axis_length

def predict_real_minor_length(minor_axis_length, mean_depth):
    input_data = pd.DataFrame([[minor_axis_length, mean_depth]], columns=['minor_axis_length', 'mean_depth'])
    real_minor_axis_length = model_minor.predict(input_data)[0]
    return real_minor_axis_length

res_major = predict_real_major_length(5000, 500)
res_minor = predict_real_minor_length(50, 500)

print(f'Predicted real major axis length: {res_major}')
print(f'Predicted real minor axis length: {res_minor}')
