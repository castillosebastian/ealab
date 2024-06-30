import pandas as pd
import numpy as np
import os 
from scipy.io import arff

# Cargar el dataset

train_file_path = os.path.join('data', 'leukemia_train_38x7129.arff')
tra, _ = arff.loadarff(train_file_path)
data = pd.DataFrame(tra)
data = data.apply(lambda x: x.decode() if isinstance(x, bytes) else x)

# Drop the 'CLASS' column if it exists
if 'CLASS' in data.columns:
    df = data.drop(columns='CLASS')


# Calcular la matriz de correlación
correlation_matrix = df.corr()

# Obtener los pares de correlaciones significativas
significant_correlations = np.where((correlation_matrix > 0.7) | (correlation_matrix < -0.7))

# Contar las correlaciones significativas, excluyendo la diagonal
significant_pairs = [(correlation_matrix.index[i], correlation_matrix.columns[j]) 
                     for i, j in zip(*significant_correlations) if i != j]

# Eliminar duplicados (ya que la matriz de correlación es simétrica)
significant_pairs = list(set([tuple(sorted(pair)) for pair in significant_pairs]))

# Número total de características
num_features = df.shape[1]

# Número total de pares posibles de características
total_pairs = (num_features * (num_features - 1)) / 2

# Número de correlaciones significativas
num_significant = len(significant_pairs)

# Calcular el porcentaje de correlaciones significativas
percentage_significant = (num_significant / total_pairs) * 100

# Mostrar el resultado
print(f'Número de correlaciones significativas: {num_significant}')
print(f'Número total de pares posibles de características: {total_pairs}')
print(f'Porcentaje de correlaciones significativas: {percentage_significant:.2f}%')