from sklearn.datasets import make_blobs
import pandas as pd

# Generar datos con 3 clusters
X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std = 2.5, random_state=42)

# Crear un DataFrame con los datos y las etiquetas
df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
df['label'] = y

# Guardar el dataset en un archivo CSV (opcional)
df.to_csv('blobs_dataset.csv', index=False)

print(df.head())

