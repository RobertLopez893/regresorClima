import pandas as pd

# Cargar el dataset original
df = pd.read_csv("Data/final_escom_data.csv")

# Filtrar los años 2022 a 2025
df_filtrado = df[(df["anio_actual"] >= 2023) & (df["anio_actual"] <= 2025)]

# Guardar el resultado en un nuevo archivo
df_filtrado.to_csv("Data/escom_datos_2023_2025.csv", index=False)

print("✅ Dataset filtrado y guardado como 'datos_2023_2025.csv'")
