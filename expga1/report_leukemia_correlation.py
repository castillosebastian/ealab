import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import os

# Load the data
train_file_path = os.path.join('/root/ealab/data', 'leukemia_train_38x7129.arff')
tra, _ = arff.loadarff(train_file_path)
data = pd.DataFrame(tra)
data = data.apply(lambda x: x.decode() if isinstance(x, bytes) else x)

# Drop the 'CLASS' column if it exists
if 'CLASS' in data.columns:
    data = data.drop(columns='CLASS')

# Calculate the correlation matrix
corr_matrix = data.corr()

# Define a high correlation threshold
threshold = 0.7

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Create a masked correlation matrix where values below the threshold are set to zero
high_corr_matrix = corr_matrix.where(np.abs(corr_matrix) >= threshold, 0)

# Plotting the high correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(high_corr_matrix, annot=False, mask=mask, cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Highly Correlated Features (Above 70%)')
plt.savefig('highly_correlated_features.png', dpi=300)
plt.show()

# Find pairs of highly correlated features excluding self-correlation
high_corr_pairs = np.where((np.abs(corr_matrix) > threshold) & (np.eye(corr_matrix.shape[0]) == 0))

# Create a DataFrame to hold highly correlated pairs
high_corr_df = pd.DataFrame({
    'Feature 1': corr_matrix.index[high_corr_pairs[0]],
    'Feature 2': corr_matrix.columns[high_corr_pairs[1]]
})
high_corr_df['Correlation'] = [corr_matrix.iat[i, j] for i, j in zip(high_corr_pairs[0], high_corr_pairs[1])]

# Calculate stats
total_high_corr = high_corr_df.shape[0] // 2  # Dividing by 2 to adjust for pairs counted twice
percentage_of_feature_space = total_high_corr / ((corr_matrix.shape[0] * (corr_matrix.shape[0] - 1)) / 2) * 100

# Write stats to a text file
with open('correlation_stats.txt', 'w') as f:
    f.write(f"Number of variables with correlation above 70%: {total_high_corr}\n")
    f.write(f"Percentage of the feature space with correlation above 70%: {percentage_of_feature_space:.2f}%\n")

# Plotting the correlation matrix without labels
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, xticklabels=False, yticklabels=False, cmap='coolwarm')
# plt.title('Correlation Matrix Visualization')
# plt.savefig('correlation_matrix.png', dpi=300)  # Adjust dpi for higher or lower resolution
# plt.show()
