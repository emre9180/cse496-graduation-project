import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# JSON dosyasını yükle
with open('merged_json.json', 'r') as file:
    data = json.load(file)

# Gerçek uzunlukları ve tahmin edilen uzunlukları saklayacak listeler
real_lengths_major = []
predicted_lengths_major = []
real_lengths_minor = []
predicted_lengths_minor = []

# Oran hesaplama fonksiyonları
def calculate_major_length_ratio(entry):
    return entry['real_major_axis_length'] / (entry['major_axis_length'] * entry['mean_depth'])

def calculate_minor_length_ratio(entry):
    return entry['real_minor_axis_length'] / (entry['minor_axis_length'] * entry['mean_depth'])

# Entry başına oranları hesapla ve tahmin edilen uzunlukları bul
for entry in data:
    ratio_major = calculate_major_length_ratio(entry)
    predicted_length_major = ratio_major * entry['major_axis_length'] * entry['mean_depth']
    real_lengths_major.append(entry['real_major_axis_length'])
    predicted_lengths_major.append(predicted_length_major)
    
    ratio_minor = calculate_minor_length_ratio(entry)
    predicted_length_minor = ratio_minor * entry['minor_axis_length'] * entry['mean_depth']
    real_lengths_minor.append(entry['real_minor_axis_length'])
    predicted_lengths_minor.append(predicted_length_minor)

# MSE hesapla
mse_major = mean_squared_error(real_lengths_major, predicted_lengths_major)
mse_minor = mean_squared_error(real_lengths_minor, predicted_lengths_minor)
print(f"Mean Squared Error for Major Axis: {mse_major}")
print(f"Mean Squared Error for Minor Axis: {mse_minor}")

# Histogram çiz
plt.figure(figsize=(10, 6))
plt.hist(predicted_lengths_major, bins=30, alpha=0.7, label='Predicted Major Lengths')
plt.hist(real_lengths_major, bins=30, alpha=0.7, label='Real Major Lengths')
plt.legend(loc='upper right')
plt.xlabel('Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Real vs Predicted Major Axis Lengths')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(predicted_lengths_minor, bins=30, alpha=0.7, label='Predicted Minor Lengths')
plt.hist(real_lengths_minor, bins=30, alpha=0.7, label='Real Minor Lengths')
plt.legend(loc='upper right')
plt.xlabel('Length (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Real vs Predicted Minor Axis Lengths')
plt.show()

# Tüm entryler için MSE hesaplamak için her bir entry başına ayrı ayrı oranlar bul ve en iyi oranı belirle
mse_values_major = []
ratios_major = []
mse_values_minor = []
ratios_minor = []

for entry in data:
    ratio_major = calculate_major_length_ratio(entry)
    ratios_major.append(ratio_major)
    predicted_lengths_major = [ratio_major * e['major_axis_length'] * e['mean_depth'] for e in data]
    mse_major = mean_squared_error([e['real_major_axis_length'] for e in data], predicted_lengths_major)
    mse_values_major.append(mse_major)
    
    ratio_minor = calculate_minor_length_ratio(entry)
    ratios_minor.append(ratio_minor)
    predicted_lengths_minor = [ratio_minor * e['minor_axis_length'] * e['mean_depth'] for e in data]
    mse_minor = mean_squared_error([e['real_minor_axis_length'] for e in data], predicted_lengths_minor)
    mse_values_minor.append(mse_minor)

# En iyi oranı bul
best_mse_index_major = np.argmin(mse_values_major)
best_ratio_major = ratios_major[best_mse_index_major]
print(f"Best Ratio for Major Axis: {best_ratio_major}")
print(f"Best MSE for Major Axis: {mse_values_major[best_mse_index_major]}")

best_mse_index_minor = np.argmin(mse_values_minor)
best_ratio_minor = ratios_minor[best_mse_index_minor]
print(f"Best Ratio for Minor Axis: {best_ratio_minor}")
print(f"Best MSE for Minor Axis: {mse_values_minor[best_mse_index_minor]}")

# En iyi oran ile tüm verileri test et
best_predicted_lengths_major = [best_ratio_major * e['major_axis_length'] * e['mean_depth'] for e in data]
best_predicted_lengths_minor = [best_ratio_minor * e['minor_axis_length'] * e['mean_depth'] for e in data]

# MSE ve oranları birlikte görselleştirmek için scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(ratios_major, mse_values_major, alpha=0.7)
plt.xlabel('Found Ratio for Major Axis')
plt.ylabel('MSE')
plt.title('Scatter Plot of Found Ratio vs MSE for Major Axis')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(ratios_minor, mse_values_minor, alpha=0.7)
plt.xlabel('Found Ratio for Minor Axis')
plt.ylabel('MSE')
plt.title('Scatter Plot of Found Ratio vs MSE for Minor Axis')
plt.show()

# En kötü oranlara sahip olanları göster
worst_mse_indices_major = np.argsort(mse_values_major)[-50:]  # En yüksek MSE'ye sahip olan ilk 50
worst_ratios_major = [ratios_major[i] for i in worst_mse_indices_major]
worst_mse_values_major = [mse_values_major[i] for i in worst_mse_indices_major]
worst_ids_major = [data[i]['id'] for i in worst_mse_indices_major]  # En kötü oranlara sahip olanların id'leri

worst_mse_indices_minor = np.argsort(mse_values_minor)[-50:]  # En yüksek MSE'ye sahip olan ilk 50
worst_ratios_minor = [ratios_minor[i] for i in worst_mse_indices_minor]
worst_mse_values_minor = [mse_values_minor[i] for i in worst_mse_indices_minor]
worst_ids_minor = [data[i]['id'] for i in worst_mse_indices_minor]  # En kötü oranlara sahip olanların id'leri

# En kötü oranlara sahip olanları yazdır
print("Top 50 entries with the worst ratios for Major Axis:")
for i in range(50):
    print(f"Entry ID: {worst_ids_major[i]} - Ratio: {worst_ratios_major[i]}, MSE: {worst_mse_values_major[i]}")

print("Top 50 entries with the worst ratios for Minor Axis:")
for i in range(50):
    print(f"Entry ID: {worst_ids_minor[i]} - Ratio: {worst_ratios_minor[i]}, MSE: {worst_mse_values_minor[i]}")

# En kötü oranları scatter plot ile görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(worst_ratios_major, worst_mse_values_major, color='red', alpha=0.7)
plt.xlabel('Worst Found Ratios for Major Axis')
plt.ylabel('MSE')
plt.title('Scatter Plot of Worst Found Ratios vs MSE for Major Axis')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(worst_ratios_minor, worst_mse_values_minor, color='red', alpha=0.7)
plt.xlabel('Worst Found Ratios for Minor Axis')
plt.ylabel('MSE')
plt.title('Scatter Plot of Worst Found Ratios vs MSE for Minor Axis')
plt.show()

# MSE aralıklarına göre kaç tane bulunan oran olduğunu gösteren bar grafiği (Major)
mse_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
mse_bin_counts_major = np.histogram(mse_values_major, bins=mse_bins)[0]

plt.figure(figsize=(10, 6))
plt.bar(['0-0.05', '0.05-0.1', '0.1-0.15', '0.15-0.2', '0.2-0.25', '0.25-0.3', '0.3-0.35', '0.35-0.4', '0.4-0.45', '0.45-0.5', '0.5-0.55'], mse_bin_counts_major, alpha=0.7)
plt.xlabel('MSE Range for Major Axis')
plt.ylabel('Number of Found Ratios')
plt.title('Number of Found Ratios in Each MSE Range for Major Axis')
plt.show()

# MSE aralıklarına göre kaç tane bulunan oran olduğunu gösteren bar grafiği (Minor)
mse_bin_counts_minor = np.histogram(mse_values_minor, bins=mse_bins)[0]

plt.figure(figsize=(10, 6))
plt.bar(['0-0.05', '0.05-0.1', '0.1-0.15', '0.15-0.2', '0.2-0.25', '0.25-0.3', '0.3-0.35', '0.35-0.4', '0.4-0.45', '0.45-0.5', '0.5-0.55'], mse_bin_counts_minor, alpha=0.7)
plt.xlabel('MSE Range for Minor Axis')
plt.ylabel('Number of Found Ratios')
plt.title('Number of Found Ratios in Each MSE Range for Minor Axis')
plt.show()
