import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Örnek veri seti (veri setinizi buraya yükleyin)
data = {
    'pixel_major_axis': [65, 69, 75, 52, 47, 49, 43.37, 58, 54, 59, 43, 63],
    'depth': [642, 541, 650, 745, 851, 871, 877.77, 719, 892, 695, 894, 732],
    'real_major_axis': [3.4, 3.4, 3.7, 3.5, 3.6, 4.3, 3.3, 3.3, 3.7, 3.5, 3.4, 4.3]
}

# Veriyi bir DataFrame'e dönüştürme
df = pd.DataFrame(data)

# Bağımsız ve bağımlı değişkenleri ayırma
X = df[['pixel_major_axis', 'depth']]
y = df['real_major_axis']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Modelin performansını değerlendirme
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)

# Modelin katsayılarını yazdırma
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Tahminler
for i in range(len(X_test)):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test.values[i]}")

# Polynomial regression with degree 2
poly = PolynomialFeatures(degree=2)
model = make_pipeline(poly, LinearRegression())

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model.fit(X_train, y_train)

# Modelin performansını değerlendirme
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)

# Tahminler
for i in range(len(X_test)):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test.values[i]}")