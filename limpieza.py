# Parte 2: Limpieza final
import pandas as pd

# Original de Parte 1
df = pd.read_csv("Data/cleaned11.csv")

if 'hora' in df.columns:
    df['fecha'] = pd.to_datetime(df['fecha'] + ' ' + df['hora'])
    df = df.drop(columns=['hora'])
else:
    df['fecha'] = pd.to_datetime(df['fecha'])

temp_col = 'temp' if 'temp' in df.columns else 'temperatura'

df = df.sort_values('fecha').reset_index(drop=True)

df['dia_anio'] = df['fecha'].dt.dayofyear
df['mes_actual'] = df['fecha'].dt.month
df['anio_actual'] = df['fecha'].dt.year
df['hora_actual'] = df['fecha'].dt.hour

df = df.set_index('fecha')
df[f'{temp_col}_hace1hora']     = df[temp_col].shift(1)
df[f'{temp_col}_hace3horas']    = df[temp_col].shift(3)
df[f'{temp_col}_hace6horas']    = df[temp_col].shift(6)
df[f'{temp_col}_hace12horas']   = df[temp_col].shift(12)
df[f'{temp_col}_hace1dia']      = df[temp_col].shift(24)
df[f'{temp_col}_hace3dias']     = df[temp_col].shift(72)
df[f'{temp_col}_hace5dias']     = df[temp_col].shift(120)
df[f'{temp_col}_hace1semana']   = df[temp_col].shift(168)

# --- ¡AQUÍ ESTÁ LA CORRECCIÓN! ---
df[f'{temp_col}_hace1mes']      = df[temp_col].shift(730)  # 730 horas ≈ 1 mes
df[f'{temp_col}_hace1anio']     = df[temp_col].shift(8760) # 8760 horas = 1 año
# --- FIN DE LA CORRECCIÓN ---

df = df.reset_index()

# Guardamos con coma (estándar)
df.to_csv("Data/2clean11.csv", index=False, sep=',')
print("Parte 2 (Limpieza) finalizada y corregida.")
