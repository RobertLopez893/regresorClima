# Práctica 8: Regresor de Clima
"""
Integrantes:
González Martínez Silvia
López Reyes José Roberto
"""

# Parte 2: Limpieza final
import pandas as pd

nombre_archivo_original = 'Data/final_datset_clima.csv'

print(f"Cargando el archivo: {nombre_archivo_original}.")
try:
    df = pd.read_csv(nombre_archivo_original)
except FileNotFoundError:
    print(f"No se encontró el archivo '{nombre_archivo_original}'.")
    exit()

# Eliminar datos vacíos
df_sin_nulos = df.dropna()

# Eliminar columna fecha
df_final = df_sin_nulos.drop('fecha', axis=1)

# Dataset super final
nombre_archivo_final = 'super_final_dataset.csv'
df_final.to_csv(nombre_archivo_final, index=False)

print(f"\nDataset final guardado como: '{nombre_archivo_final}'")
print(df_final)
