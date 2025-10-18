import pandas as pd

nombre_archivo_original = 'Data/Sierra México, Tlalpexco,... 2024-10-01 to 2025-10-16.csv'

print(f"Cargando el archivo: {nombre_archivo_original}.")
df = pd.read_csv(nombre_archivo_original)

df_limpio = df[['datetime', 'temp']].copy()
df_limpio['datetime'] = pd.to_datetime(df_limpio['datetime'])

df_limpio['fecha'] = df_limpio['datetime'].dt.date
df_limpio['hora'] = df_limpio['datetime'].dt.time

df_final = df_limpio[['fecha', 'hora', 'temp']]

nombre_archivo_final = 'final_dataset.csv'
df_final.to_csv(nombre_archivo_final, index=False)

print(f"\nDataset guardado como: {nombre_archivo_final}\n")
print(df_final)
