# Práctica 8: Regresor de Clima
"""
Integrantes:
González Martínez Silvia
López Reyes José Roberto
"""

# Parte 5: Probando predicción recursiva
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
import warnings

# Ignoramos los Warnings
warnings.filterwarnings('ignore')

# Configuración Inicial
RUTA_DATOS = 'Data/data_upto_10_nov.csv'
COLUMNA_TARGET = 'temp'

# Cargar y construir el índice de fechas
try:
    # Cargar el CSV sin procesar fechas todavía
    df = pd.read_csv(RUTA_DATOS)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta: {RUTA_DATOS}")
    exit()

try:
    # Construir la fecha a partir del año y el día del año
    fechas = pd.to_datetime(
        df['anio_actual'].astype(str) + df['dia_anio'].astype(str),
        format='%Y%j'
    )

    # Añadir las horas a esas fechas
    fechas_con_hora = fechas + pd.to_timedelta(df['hora_actual'], unit='H')

    # Establecer esto como el nuevo índice del DataFrame
    df.index = fechas_con_hora

except KeyError as e:
    print(f"Error: No se encontró la columna '{e}' en el CSV.")
    exit()
except Exception as e:
    print(f"Error inesperado al construir la fecha: {e}")
    exit()

# Limpiar, ordenar por el nuevo índice y eliminar duplicados
df = df.dropna().drop_duplicates().sort_index()

# Feature Engineering

# Hora (0-23, 24 valores)
df['hora_sin'] = np.sin(2 * np.pi * df['hora_actual'] / 24.0)
df['hora_cos'] = np.cos(2 * np.pi * df['hora_actual'] / 24.0)

# Mes (1-12, 12 valores)
df['mes_sin'] = np.sin(2 * np.pi * df['mes_actual'] / 12.0)
df['mes_cos'] = np.cos(2 * np.pi * df['mes_actual'] / 12.0)

# Día del año (1-365)
df['dia_anio_sin'] = np.sin(2 * np.pi * df['dia_anio'] / 365.0)
df['dia_anio_cos'] = np.cos(2 * np.pi * df['dia_anio'] / 365.0)

# Lags de Horas
df['temperatura_hace1hora'] = df[COLUMNA_TARGET].shift(1)
df['temperatura_hace3horas'] = df[COLUMNA_TARGET].shift(3)
df['temperatura_hace6horas'] = df[COLUMNA_TARGET].shift(6)
df['temperatura_hace12horas'] = df[COLUMNA_TARGET].shift(12)

# Lags de Días
df['temperatura_hace1dia'] = df[COLUMNA_TARGET].shift(24)
df['temperatura_hace3dias'] = df[COLUMNA_TARGET].shift(24 * 3)
df['temperatura_hace1semana'] = df[COLUMNA_TARGET].shift(24 * 7)

# Lags de Año
df['temperatura_hace1anio'] = df[COLUMNA_TARGET].shift(365 * 24) # Asumiendo datos completos

# Features de ventana móvil (rolling)
# Media de las últimas 3 horas
df['temp_media_ultimas_3h'] = df[COLUMNA_TARGET].shift(1).rolling(window=3).mean()
# Media de las últimas 24 horas
df['temp_media_ultimas_24h'] = df[COLUMNA_TARGET].shift(1).rolling(window=24).mean()

# Eliminamos las columnas originales que ya no necesitamos
df = df.drop(columns=['hora_actual', 'mes_actual', 'dia_anio'])
df = df.dropna()

# Separar variables
X = df.drop(COLUMNA_TARGET, axis=1)

y = df[COLUMNA_TARGET]

# Guardamos los nombres de las columnas en el orden correcto
COLUMNAS_FEATURES_ORDENADAS = X.columns.tolist()

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Solo se transforma

# Tuneando el modelo
parameters = {
    'max_iter': [200, 400, 600],  # Más árboles
    'learning_rate': [0.05, 0.1],
    'max_depth': [7, 10, 15],     # Árboles más profundos
    'l2_regularization': [0.5, 1.0],
    'min_samples_leaf': [20, 40]
}

# Usando TSS
tscv = TimeSeriesSplit(n_splits=5)

# Configurar GridSearchCV
gb = HistGradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(gb, parameters, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Obtener mejores parámetros
best_params = grid_search.best_params_
print(f"Mejores parámetros encontrados: {best_params}")

# Evaluar modelo óptimo en el conjunto de Test
best_model_eval = grid_search.best_estimator_
y_pred = best_model_eval.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluación del Modelo (sobre datos de Test) ---")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Visualización de predicción vs. real
plt.figure(figsize=(15, 6))
plt.plot(y_test.index, y_test.values, label='Real', color='blue', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predicho (Test)', color='orange', linestyle='--')
plt.title('Evaluación del Modelo: Predicción vs. Real (Test Set)')
plt.xlabel('Fecha')
plt.ylabel('Temperatura')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_vs_pred.png")
plt.show()

# Re-entrenar el escalador con todos los datos
final_scaler = StandardScaler()
X_full_scaled = final_scaler.fit_transform(X)

# Re-entrenar el modelo con los mejores parámetros y todos los datos
final_model = HistGradientBoostingRegressor(**best_params, random_state=42)
final_model.fit(X_full_scaled, y)


def predecir_futuro_recursivo(modelo, scaler, historial_df, pasos_a_predecir, freq='H'):
    """
    Predice el futuro paso a paso, usando predicciones anteriores
    para alimentar los lags.
    """

    # Copiamos el historial para no modificar el original
    df_predicciones = historial_df.copy()

    # Obtenemos la última fecha real y generamos las fechas futuras
    ultima_fecha_real = df_predicciones.index.max()

    # Determinamos el paso basado en la frecuencia
    if freq == 'H':
        siguiente_paso = pd.Timedelta(hours=1)
    else:
        siguiente_paso = pd.Timedelta(days=1)

    # Creamos el date_range usando ese paso
    fechas_futuras = pd.date_range(
        start=ultima_fecha_real + siguiente_paso,
        periods=pasos_a_predecir,
        freq=freq
    )

    # Bucle principal de predicción
    for fecha_actual in fechas_futuras:

        # Crear el vector de features para esta 'fecha_actual'
        nueva_fila_features = {}

        # --- Features Cíclicas y de Calendario (Sin cambios) ---
        if 'hora_sin' in COLUMNAS_FEATURES_ORDENADAS:
            nueva_fila_features['hora_sin'] = np.sin(2 * np.pi * fecha_actual.hour / 24.0)
        if 'hora_cos' in COLUMNAS_FEATURES_ORDENADAS:
            nueva_fila_features['hora_cos'] = np.cos(2 * np.pi * fecha_actual.hour / 24.0)
        if 'mes_sin' in COLUMNAS_FEATURES_ORDENADAS:
            nueva_fila_features['mes_sin'] = np.sin(2 * np.pi * fecha_actual.month / 12.0)
        if 'mes_cos' in COLUMNAS_FEATURES_ORDENADAS:
            nueva_fila_features['mes_cos'] = np.cos(2 * np.pi * fecha_actual.month / 12.0)
        if 'dia_anio_sin' in COLUMNAS_FEATURES_ORDENADAS:
            nueva_fila_features['dia_anio_sin'] = np.sin(2 * np.pi * fecha_actual.dayofyear / 365.0)
        if 'dia_anio_cos' in COLUMNAS_FEATURES_ORDENADAS:
            nueva_fila_features['dia_anio_cos'] = np.cos(2 * np.pi * fecha_actual.dayofyear / 365.0)
        if 'anio_actual' in COLUMNAS_FEATURES_ORDENADAS:
            nueva_fila_features['anio_actual'] = fecha_actual.year

        # --- Rellenar features de lag y rolling ---
        try:
            # Lags de Horas
            if 'temperatura_hace1hora' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace1hora'] = \
                    df_predicciones.loc[fecha_actual - pd.Timedelta(hours=1)][COLUMNA_TARGET]
            if 'temperatura_hace3horas' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace3horas'] = \
                    df_predicciones.loc[fecha_actual - pd.Timedelta(hours=3)][COLUMNA_TARGET]
            if 'temperatura_hace6horas' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace6horas'] = \
                    df_predicciones.loc[fecha_actual - pd.Timedelta(hours=6)][COLUMNA_TARGET]
            if 'temperatura_hace12horas' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace12horas'] = \
                    df_predicciones.loc[fecha_actual - pd.Timedelta(hours=12)][COLUMNA_TARGET]

            # Lags de Días
            if 'temperatura_hace1dia' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace1dia'] = df_predicciones.loc[fecha_actual - pd.Timedelta(days=1)][
                    COLUMNA_TARGET]
            if 'temperatura_hace3dias' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace3dias'] = df_predicciones.loc[fecha_actual - pd.Timedelta(days=3)][
                    COLUMNA_TARGET]

            # --- NOTA: Tu script original no incluía este lag en la función ---
            if 'temperatura_hace1semana' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace1semana'] = \
                    df_predicciones.loc[fecha_actual - pd.Timedelta(days=7)][COLUMNA_TARGET]

            # Lags de Mes y Año
            # --- NOTA: Tu script original no incluía estos lags en la función ---
            if 'temperatura_hace1mes' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace1mes'] = df_predicciones.loc[fecha_actual - DateOffset(months=1)][
                    COLUMNA_TARGET]
            if 'temperatura_hace1anio' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace1anio'] = df_predicciones.loc[fecha_actual - DateOffset(years=1)][
                    COLUMNA_TARGET]

            # --- BLOQUE AÑADIDO: Cálculo de Medias Móviles ---
            if 'temp_media_ultimas_3h' in COLUMNAS_FEATURES_ORDENADAS:
                # Calcula la media de las 3 horas anteriores
                media_3h = df_predicciones.loc[
                           fecha_actual - pd.Timedelta(hours=3): fecha_actual - pd.Timedelta(hours=1)
                           ][COLUMNA_TARGET].mean()
                nueva_fila_features['temp_media_ultimas_3h'] = media_3h

            if 'temp_media_ultimas_24h' in COLUMNAS_FEATURES_ORDENADAS:
                # Calcula la media de las 24 horas anteriores
                media_24h = df_predicciones.loc[
                            fecha_actual - pd.Timedelta(hours=24): fecha_actual - pd.Timedelta(hours=1)
                            ][COLUMNA_TARGET].mean()
                nueva_fila_features['temp_media_ultimas_24h'] = media_24h
            # --- FIN DE BLOQUE AÑADIDO ---

        except KeyError as e:
            # Esto puede pasar si un lag (ej. 1 año) no se encuentra
            print(f"Advertencia: No se encontró el lag para la fecha: {e}. Se usará 'None'.")
            # Continuamos, pero 'nueva_fila_features' tendrá 'None' para la clave que falló
        except Exception as e:
            print(f"Error irrecuperable con fecha: {fecha_actual}, Error: {e}")
            return None

        # --- PARCHE PARA NaNs ---
        # Si algún cálculo (ej. media móvil al inicio) da NaN,
        # lo rellenamos con el valor más reciente (1 hora antes)
        for key, value in nueva_fila_features.items():
            if pd.isna(value):
                try:
                    fallback_value = df_predicciones.loc[fecha_actual - pd.Timedelta(hours=1)][COLUMNA_TARGET]
                    nueva_fila_features[key] = fallback_value
                    print(f"Advertencia: Se rellenó NaN para '{key}' en fecha {fecha_actual}")
                except Exception as e_fallback:
                    print(f"Error crítico: No se pudo rellenar NaN ni encontrar fallback para {key}: {e_fallback}")
                    return None
        # --- FIN DE PARCHE ---

        # Convertir la fila de features a un DataFrame
        X_unscaled_row = pd.DataFrame(nueva_fila_features, index=[fecha_actual], columns=COLUMNAS_FEATURES_ORDENADAS)

        # Escalar la nueva fila de features
        X_scaled_row = scaler.transform(X_unscaled_row)

        # Predecir temperatura
        prediccion = modelo.predict(X_scaled_row)[0]

        # Guardar la predicción
        df_predicciones.loc[fecha_actual, COLUMNA_TARGET] = prediccion

        # Añadimos a las otras columnas
        for col, val in nueva_fila_features.items():
            df_predicciones.loc[fecha_actual, col] = val

    # Devolver solo las predicciones nuevas
    return df_predicciones.loc[fechas_futuras]


# Obtenemos la fecha/hora actual
ahora = pd.Timestamp.now().tz_localize(None)

# Obtenemos la última fecha de nuestros datos
ultima_fecha_real = df.index.max()

# Calculamos cuántas horas hay entre el último dato y ahora
# Sumamos 24 horas extra como colchón, para poder ver el pronóstico de mañana
horas_necesarias = (ahora - ultima_fecha_real).total_seconds() / 3600
PASOS_FUTUROS = int(np.ceil(horas_necesarias)) + 24  # ¡Ya no es 168!

print(f"\nÚltimo dato en el historial: {ultima_fecha_real}")
print(f"Fecha/hora actual:         {ahora}")

predicciones_futuras = predecir_futuro_recursivo(
    final_model,
    final_scaler,
    df,
    pasos_a_predecir=PASOS_FUTUROS,
    freq='H'
)

if predicciones_futuras is not None:
    # Redondeamos la hora actual a la hora en punto más cercana
    ahora_redondeado = ahora.floor('H')

    prediccion_actual = 0.0

    try:
        # Buscamos la predicción para esta hora exacta
        prediccion_actual = predicciones_futuras.loc[ahora_redondeado][COLUMNA_TARGET]
        print("\n" + "=" * 50)
        print(f"🌤️ PREDICCIÓN PARA EL MOMENTO ACTUAL ({ahora_redondeado}) 🌤️")
        print(f"   Temperatura estimada: {prediccion_actual:.2f}°C")
        print("=" * 50 + "\n")

    except KeyError:
        print(f"No se pudo encontrar la predicción para {ahora_redondeado} en los resultados.")

    # Visualización final (Historial + Futuro)
    plt.figure(figsize=(15, 7))

    # Graficar los últimos 10 días de datos reales
    plt.plot(
        df.last('10D').index,
        df.last('10D')[COLUMNA_TARGET],
        label='Historial Reciente',
        color='blue'
    )

    # Graficar la predicción futura
    plt.plot(
        predicciones_futuras.index,
        predicciones_futuras[COLUMNA_TARGET],
        label='Predicción Futura (Recursiva)',
        color='red',
        linestyle='--'
    )

    # Resaltar la predicción actual en la gráfica
    try:
        plt.axvline(x=ahora_redondeado, color='green', linestyle=':', lw=2, label=f'Ahora ({prediccion_actual:.2f}°C)')
    except:
        pass

    plt.title('Predicción Futura de Temperatura')
    plt.xlabel('Fecha')
    plt.ylabel('Temperatura')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Guardar las predicciones en un CSV
    try:
        predicciones_futuras.to_csv("predicciones_futuras.csv")
        print("\nPredicciones guardadas en 'predicciones_futuras.csv'")
    except Exception as e:
        print(f"\nNo se pudieron guardar las predicciones: {e}")
else:
    print("\nNo se pudieron generar las predicciones futuras debido a un error.")