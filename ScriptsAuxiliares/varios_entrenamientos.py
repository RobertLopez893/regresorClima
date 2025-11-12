# Práctica 8: Regresor de Clima
"""
Integrantes:
González Martínez Silvia
López Reyes José Roberto
"""

# Parte 5: ¡¡LA ARENA DE HIPERPARÁMETROS!!
import pandas as pd
import numpy as np
import time

# --- IMPORTS DE TODOS LOS MODELOS ---
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor  # <--- Añadido el "Rey"
# --- FIN DE IMPORTS ---

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import DateOffset
import warnings

# Ignoramos los Warnings
warnings.filterwarnings('ignore')

# Configuración Inicial
RUTA_DATOS = 'Data/escom_datos_2023_2025.csv'
COLUMNA_TARGET = 'temp'

# Cargar y construir el índice de fechas
try:
    df = pd.read_csv(RUTA_DATOS)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta: {RUTA_DATOS}")
    exit()

try:
    fechas = pd.to_datetime(
        df['anio_actual'].astype(str) + df['dia_anio'].astype(str),
        format='%Y%j'
    )
    fechas_con_hora = fechas + pd.to_timedelta(df['hora_actual'], unit='H')
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
print("Iniciando Feature Engineering...")
df['hora_sin'] = np.sin(2 * np.pi * df['hora_actual'] / 24.0)
df['hora_cos'] = np.cos(2 * np.pi * df['hora_actual'] / 24.0)
df['mes_sin'] = np.sin(2 * np.pi * df['mes_actual'] / 12.0)
df['mes_cos'] = np.cos(2 * np.pi * df['mes_actual'] / 12.0)
df['dia_anio_sin'] = np.sin(2 * np.pi * df['dia_anio'] / 365.0)
df['dia_anio_cos'] = np.cos(2 * np.pi * df['dia_anio'] / 365.0)
df['temperatura_hace1hora'] = df[COLUMNA_TARGET].shift(1)
df['temperatura_hace3horas'] = df[COLUMNA_TARGET].shift(3)
df['temperatura_hace6horas'] = df[COLUMNA_TARGET].shift(6)
df['temperatura_hace12horas'] = df[COLUMNA_TARGET].shift(12)
df['temperatura_hace1dia'] = df[COLUMNA_TARGET].shift(24)
df['temperatura_hace3dias'] = df[COLUMNA_TARGET].shift(24 * 3)
df['temperatura_hace1semana'] = df[COLUMNA_TARGET].shift(24 * 7)
df['temperatura_hace1anio'] = df[COLUMNA_TARGET].shift(365 * 24)
df['temp_media_ultimas_3h'] = df[COLUMNA_TARGET].shift(1).rolling(window=3).mean()
df['temp_media_ultimas_24h'] = df[COLUMNA_TARGET].shift(1).rolling(window=24).mean()
df = df.drop(columns=['hora_actual', 'mes_actual', 'dia_anio'])
df = df.dropna()
print("Feature Engineering completado.")

# Separar variables
X = df.drop(COLUMNA_TARGET, axis=1)
y = df[COLUMNA_TARGET]

# Guardamos los nombres de las columnas en el orden correcto
COLUMNAS_FEATURES_ORDENADAS = X.columns.tolist()

# División de datos (PARA EL GRIDSEARCH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Escalado (Lo necesitamos para TODOS los modelos)
print("Preparando escaladores...")
# Escalador solo para el GridSearch (entrenado en Train)
train_scaler = StandardScaler()
X_train_scaled = train_scaler.fit_transform(X_train)

# Escalador para los modelos finales
final_scaler = StandardScaler()
X_full_scaled = final_scaler.fit_transform(X)

# --- NUEVA SECCIÓN DE ENTRENAMIENTO MASIVO ---

# Usando TSS
tscv = TimeSeriesSplit(n_splits=5)

# 1. Definir los modelos y sus cuadrículas de parámetros
# (Reducidos para que no tarde 3 días)
print("Definiendo la arena de modelos y sus grids...")
models_and_grids = {
    "HistGradientBoostingRegressor": (
        HistGradientBoostingRegressor(random_state=42),
        {
            'max_iter': [200, 400],
            'learning_rate': [0.05, 0.1],
            'max_depth': [7, 10]
        }
    ),
    "LGBMRegressor": (
        LGBMRegressor(random_state=42, n_jobs=-1),
        {
            'n_estimators': [200, 400],
            'learning_rate': [0.05, 0.1],
            'max_depth': [7, 10]
        }
    ),
    "RandomForestRegressor": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            'n_estimators': [100, 200],
            'max_depth': [10, 15],
            'min_samples_leaf': [20, 40]
        }
    ),
    "SVR": (
        SVR(),
        {
            'C': [1, 10],
            'gamma': ['scale', 'auto']
        }
    ),
    "MLPRegressor": (
        MLPRegressor(random_state=42, max_iter=500, early_stopping=True),
        {
            'hidden_layer_sizes': [(50,), (100, 50)],
            'activation': ['relu'],
            'learning_rate_init': [0.01, 0.001]
        }
    ),
    "KNeighborsRegressor": (
        KNeighborsRegressor(n_jobs=-1),
        {
            'n_neighbors': [10, 20],
            'weights': ['uniform', 'distance']
        }
    ),
    "Ridge": (
        Ridge(random_state=42),
        {
            'alpha': [0.1, 1.0, 10.0]
        }
    )
    # Nota: Omití GradientBoosting y XGBoost a propósito
    # porque son MUY lentos sin GPU. LGBM/Hist son sus reemplazos modernos.
    # Omití Lasso porque Ridge suele ser mejor.
}

# 2. Bucle de GridSearchCV + Re-entrenamiento final
print("¡¡¡INICIANDO ENTRENAMIENTO MASIVO!!!")
print("Esto va a tardar MUCHO tiempo...")
trained_models_final = {}

for model_name, (model_instance, param_grid) in models_and_grids.items():
    start_time = time.time()
    print(f"\n--- Iniciando GridSearchCV para: {model_name} ---")

    # Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=model_instance,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # ¡Entrenar el GridSearch en los datos de ENTRENAMIENTO!
    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    print(f"Mejores parámetros para {model_name}: {best_params}")

    # 3. Re-entrenar el MEJOR modelo con TODOS los datos
    print(f"Re-entrenando {model_name} con datos COMPLETOS...")

    # Creamos una nueva instancia con los mejores parámetros
    final_model = model_instance.__class__(**best_params)

    # Manejar random_state si existe en el modelo
    if 'random_state' in final_model.get_params():
        final_model.set_params(random_state=42)
    # Manejar n_jobs si existe
    if 'n_jobs' in final_model.get_params():
        final_model.set_params(n_jobs=-1)

    # ¡Entrenar en el set COMPLETO y ESCALADO!
    final_model.fit(X_full_scaled, y)

    # Guardar el modelo final
    trained_models_final[model_name] = final_model

    end_time = time.time()
    print(f"--- {model_name} completado en {((end_time - start_time) / 60):.2f} minutos ---")

print("\n\n¡¡¡ENTRENAMIENTO MASIVO COMPLETADO!!!")


# --- FIN DE SECCIÓN MODIFICADA ---


# (La función predecir_futuro_recursivo es idéntica)
def predecir_futuro_recursivo(modelo, scaler, historial_df, pasos_a_predecir, freq='H'):
    """
    Predice el futuro paso a paso, usando predicciones anteriores
    para alimentar los lags.
    """
    df_predicciones = historial_df.copy()
    ultima_fecha_real = df_predicciones.index.max()
    siguiente_paso = pd.Timedelta(hours=1)
    fechas_futuras = pd.date_range(
        start=ultima_fecha_real + siguiente_paso,
        periods=pasos_a_predecir,
        freq=freq
    )

    for fecha_actual in fechas_futuras:
        nueva_fila_features = {}
        try:
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
            if 'temperatura_hace1dia' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace1dia'] = df_predicciones.loc[fecha_actual - pd.Timedelta(days=1)][
                    COLUMNA_TARGET]
            if 'temperatura_hace3dias' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace3dias'] = df_predicciones.loc[fecha_actual - pd.Timedelta(days=3)][
                    COLUMNA_TARGET]
            if 'temperatura_hace1semana' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace1semana'] = \
                df_predicciones.loc[fecha_actual - pd.Timedelta(days=7)][COLUMNA_TARGET]
            if 'temperatura_hace1mes' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace1mes'] = df_predicciones.loc[fecha_actual - DateOffset(months=1)][
                    COLUMNA_TARGET]
            if 'temperatura_hace1anio' in COLUMNAS_FEATURES_ORDENADAS:
                nueva_fila_features['temperatura_hace1anio'] = df_predicciones.loc[fecha_actual - DateOffset(years=1)][
                    COLUMNA_TARGET]

            if 'temp_media_ultimas_3h' in COLUMNAS_FEATURES_ORDENADAS:
                media_3h = \
                df_predicciones.loc[fecha_actual - pd.Timedelta(hours=3): fecha_actual - pd.Timedelta(hours=1)][
                    COLUMNA_TARGET].mean()
                nueva_fila_features['temp_media_ultimas_3h'] = media_3h
            if 'temp_media_ultimas_24h' in COLUMNAS_FEATURES_ORDENADAS:
                media_24h = \
                df_predicciones.loc[fecha_actual - pd.Timedelta(hours=24): fecha_actual - pd.Timedelta(hours=1)][
                    COLUMNA_TARGET].mean()
                nueva_fila_features['temp_media_ultimas_24h'] = media_24h
        except KeyError as e:
            # print(f"Advertencia: No se encontró el lag para la fecha: {e}. Se usará 'None'.")
            pass
        except Exception as e:
            print(f"Error irrecuperable con fecha: {fecha_actual}, Error: {e}")
            return None

        for key, value in nueva_fila_features.items():
            if pd.isna(value):
                try:
                    fallback_value = df_predicciones.loc[fecha_actual - pd.Timedelta(hours=1)][COLUMNA_TARGET]
                    nueva_fila_features[key] = fallback_value
                except Exception:
                    nueva_fila_features[key] = 0

        X_unscaled_row = pd.DataFrame(nueva_fila_features, index=[fecha_actual], columns=COLUMNAS_FEATURES_ORDENADAS)
        X_unscaled_row = X_unscaled_row.fillna(0)
        X_scaled_row = scaler.transform(X_unscaled_row)
        prediccion = modelo.predict(X_scaled_row)[0]
        df_predicciones.loc[fecha_actual, COLUMNA_TARGET] = prediccion
        for col, val in nueva_fila_features.items():
            df_predicciones.loc[fecha_actual, col] = val

    return df_predicciones.loc[fechas_futuras]


# --- SECCIÓN DE PREDICCIÓN FINAL MODIFICADA ---

print("\nIniciando predicciones recursivas para todos los modelos OPTIMIZADOS...")
ahora = pd.Timestamp.now().tz_localize(None)
ahora_redondeado = ahora.floor('H')
ultima_fecha_real = df.index.max()
horas_necesarias = (ahora - ultima_fecha_real).total_seconds() / 3600
PASOS_FUTUROS = int(np.ceil(horas_necesarias)) + 24

print(f"Último dato en el historial: {ultima_fecha_real}")
print(f"Prediciendo para la hora:  {ahora_redondeado}")
print(f"Se generarán {PASOS_FUTUROS} pasos recursivos por modelo...")

resultados_predicciones = {}

for model_name, model in trained_models_final.items():
    print(f"\n--- Calculando para: {model_name} (Optimizado) ---")

    predicciones_futuras = predecir_futuro_recursivo(
        model,
        final_scaler,  # Usamos el escalador final
        df,
        pasos_a_predecir=PASOS_FUTUROS,
        freq='H'
    )

    if predicciones_futuras is not None:
        try:
            prediccion_actual = predicciones_futuras.loc[ahora_redondeado][COLUMNA_TARGET]
            resultados_predicciones[model_name] = prediccion_actual
            print(f"   Predicción: {prediccion_actual:.2f}°C")
        except KeyError:
            print(f"   Error: No se pudo encontrar la predicción para {ahora_redondeado}.")
            resultados_predicciones[model_name] = np.nan
        except Exception as e:
            print(f"   Error inesperado: {e}")
            resultados_predicciones[model_name] = np.nan
    else:
        print("   Fallo en la generación de predicciones recursivas.")
        resultados_predicciones[model_name] = np.nan

# --- Imprimir el Resumen Final ---
print("\n" + "=" * 50)
print(f"🏆 COMPARATIVA DE PREDICCIONES OPTIMIZADAS (Hora: {ahora_redondeado}) 🏆")
print("=" * 50)
for model_name, prediccion in resultados_predicciones.items():
    if not pd.isna(prediccion):
        print(f"   {model_name:<30} {prediccion:>10.2f}°C")
    else:
        print(f"   {model_name:<30} {'FALLÓ':>10}")
print("=" * 50)
