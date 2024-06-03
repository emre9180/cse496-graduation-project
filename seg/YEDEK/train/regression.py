import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Read the JSON file
with open('ellipse_results.json', 'r') as file:
    data = json.load(file)

# Step 2: Extract the relevant features and target variables
df = pd.DataFrame(data)
X = df[['major_axis_length', 'minor_axis_length', 'mean_depth']]
y_major = df['real_major_axis_length']
y_minor = df['real_minor_axis_length']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_major_train, y_major_test, y_minor_train, y_minor_test = train_test_split(
    X, y_major, y_minor, test_size=0.2, random_state=42)

# Step 4: Train the regression models
model_major = LinearRegression()
model_minor = LinearRegression()

model_major.fit(X_train, y_major_train)
model_minor.fit(X_train, y_minor_train)

# Step 5: Evaluate the models
y_major_pred = model_major.predict(X_test)
y_minor_pred = model_minor.predict(X_test)

print(f'Major Axis Length Model:')
print(f'Mean squared error: {mean_squared_error(y_major_test, y_major_pred)}')
print(f'Coefficient of determination: {r2_score(y_major_test, y_major_pred)}')

print(f'Minor Axis Length Model:')
print(f'Mean squared error: {mean_squared_error(y_minor_test, y_minor_pred)}')
print(f'Coefficient of determination: {r2_score(y_minor_test, y_minor_pred)}')

# Step 6: Predict the values using the trained model
def predict_real_lengths(major_axis_length, minor_axis_length, mean_depth):
    input_data = pd.DataFrame([[major_axis_length, minor_axis_length, mean_depth]], 
                              columns=['major_axis_length', 'minor_axis_length', 'mean_depth'])
    real_major_axis_length = model_major.predict(input_data)[0]
    real_minor_axis_length = model_minor.predict(input_data)[0]
    return real_major_axis_length, real_minor_axis_length

# Example prediction
example_major, example_minor = predict_real_lengths(48.983489990234375, 42.18183517456055, 750.8571166992188)
print(f'Predicted real major axis length: {example_major}')
print(f'Predicted real minor axis length: {example_minor}')
