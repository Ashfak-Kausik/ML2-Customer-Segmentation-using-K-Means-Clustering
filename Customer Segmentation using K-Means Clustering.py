import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample customer data
data = {'CustomerID': [1, 2, 3, 4, 5],
        'Annual Income (k$)': [15, 16, 17, 18, 19],
        'Spending Score (1-100)': [39, 81, 6, 77, 40]}

df = pd.DataFrame(data)

# Prepare the data
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-Means
kmeans = KMeans(n_clusters=2)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
