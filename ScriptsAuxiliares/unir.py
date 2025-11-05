import pandas as pd

# Cargar los dos datasets
df_original = pd.read_csv("Data/the_super_final_holy_dataset.csv")     # tu dataset base
df_reciente = pd.read_csv("Data/recent_data_second.csv")          # el más nuevo

# Unirlos en el mismo orden
df_final = pd.concat([df_original, df_reciente], ignore_index=True)

# Guardar el resultado
df_final.to_csv("Data/data_upto_02nov_2020.csv", index=False)

print("✅ Datasets unidos y guardados como 'datos_2022_2025.csv'")
