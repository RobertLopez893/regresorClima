# Práctica 8: Regresor de Clima
"""
Integrantes:
González Martínez Silvia
López Reyes José Roberto
"""

# Parte 3: Entrenamiento del modelo
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

nombre_archivo_original = 'Data/escom_data2.csv'

print(f"Cargando el archivo: {nombre_archivo_original}.")
try:
    df = pd.read_csv(nombre_archivo_original)
except FileNotFoundError:
    print(f"No se encontró el archivo '{nombre_archivo_original}'.")
    exit()

print(df)

# Eliminar datos vacíos
df_sin_nulos = df.dropna()

# Eliminar columna fecha
df_final = df_sin_nulos.drop('fecha', axis=1)

# Dataset super final
nombre_archivo_final = 'Data/final_escom_data.csv'
df_final.to_csv(nombre_archivo_final, index=False)

print(f"\nDataset final guardado como: '{nombre_archivo_final}'")
print(df_final)

X = df_final.drop('temp', axis=1)
y = df_final['temp']

print(X)
print(y)

X_train, X_test = X.iloc[:int(len(X) * 0.8)], X.iloc[int(len(X) * 0.8):]
y_train, y_test = y.iloc[:int(len(X) * 0.8)], y.iloc[int(len(X) * 0.8):]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""GradientBoosting"""

gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModelo: GradientBoosting")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

"""HistGradientBoosting"""

hgb = HistGradientBoostingRegressor()
hgb.fit(X_train, y_train)
y_pred = hgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModelo: HistGradientBoosting")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)
