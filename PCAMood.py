import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('/kaggle/input/pcamood/PCAMood.csv')

# Separate the features and target variable
X = df.drop(columns=['mood'])
y = df['mood']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=len(X.columns))  # We initially take as many components as features
X_pca = pca.fit_transform(X_scaled)

# Feature importance ranking based on PCA components
explained_variance = pca.explained_variance_ratio_

# Calculate the contribution of each feature to the principal components
components_importance = np.abs(pca.components_)

# Rank features based on their importance
features_importance = np.sum(components_importance * explained_variance.reshape(-1, 1), axis=0)
features_ranked = sorted(zip(X.columns, features_importance), key=lambda x: x[1], reverse=True)

# Display feature ranking
print("Feature Ranking:")
for i, (feature, importance) in enumerate(features_ranked):
    print(f"{i+1}. {feature}: {importance}")

# Select the top 12 features by eliminating the least important one
top_12_features = [feature for feature, _ in features_ranked[:12]]
print("\nTop 12 features:")
print(top_12_features)

# Visualizing the explained variance
plt.figure(figsize=(10, 7))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center',
        label='Individual explained variance')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.show()

