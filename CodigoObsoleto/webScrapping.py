import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime
import re

# URL para el año 2023
url = 'https://es.weatherspark.com/h/y/5674/2023/Datos-hist%C3%B3ricos-meteorol%C3%B3gicos-de-2023-en-Ciudad-de-M%C3%A9xico-M%C3%A9xico'

print("--- INICIANDO DIAGNÓSTICO DE WEB SCRAPING ---")

# --- Paso 1: Realizar la solicitud HTTP ---
print("\n[Paso 1/5] Realizando la solicitud GET a la URL...")
try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8'
    }
    response = requests.get(url, headers=headers, timeout=15)  # Timeout de 15 segundos

    # raise_for_status() lanzará un error para respuestas 4xx o 5xx (ej. 404 No Encontrado, 403 Prohibido)
    response.raise_for_status()

    print(f"-> ÉXITO: Se recibió una respuesta exitosa (Código: {response.status_code})")
    html_content = response.text

except requests.exceptions.HTTPError as e:
    print(f"-> ERROR FATAL: El servidor devolvió un error. Código: {e.response.status_code}.")
    print("   Esto puede significar que la página no existe o que nuestro acceso está bloqueado.")
    exit()
except requests.exceptions.RequestException as e:
    print(f"-> ERROR FATAL: Falló la conexión a la URL. Detalles: {e}")
    print("   Verifica tu conexión a internet o si la URL es correcta.")
    exit()

# --- Paso 2: Analizar el HTML con BeautifulSoup ---
print("\n[Paso 2/5] Analizando el contenido HTML de la página...")
try:
    soup = BeautifulSoup(html_content, 'html.parser')
    print("-> ÉXITO: El HTML fue analizado correctamente.")
except Exception as e:
    print(f"-> ERROR FATAL: Falló el análisis del HTML. Detalles: {e}")
    exit()

# --- Paso 3: Encontrar la etiqueta <script> correcta ---
print("\n[Paso 3/5] Buscando la etiqueta <script> que contiene 'window.APP_STATE'...")
all_scripts = soup.find_all('script')
target_script_content = None

if not all_scripts:
    print("-> ERROR FATAL: No se encontró NINGUNA etiqueta <script> en la página.")
    exit()

for script in all_scripts:
    # A veces el contenido del script puede no ser un string
    if script.string and 'window.APP_STATE' in script.string:
        target_script_content = script.string
        break

if target_script_content:
    print("-> ÉXITO: Se encontró la etiqueta <script> con los datos.")
else:
    print("-> ERROR FATAL: Se analizaron todas las etiquetas <script> pero NINGUNA contiene 'window.APP_STATE'.")
    print("   Posible causa: La página cambió el nombre de la variable o ya no la incluye directamente en el HTML.")
    exit()

# --- Paso 4: Extraer el objeto JSON del script ---
print("\n[Paso 4/5] Extrayendo el objeto JSON del texto del script...")
try:
    # Usamos una expresión regular para encontrar el JSON que empieza con { y termina con }
    # Esto es más robusto que hacer split.
    match = re.search(r'window\.APP_STATE = (\{.*\});', target_script_content)
    if match:
        json_text = match.group(1)
        full_data = json.loads(json_text)
        print("-> ÉXITO: El objeto JSON fue extraído y decodificado.")
    else:
        print("-> ERROR FATAL: Se encontró el script, pero el formato no coincide con 'window.APP_STATE = {...};'.")
        exit()
except json.JSONDecodeError as e:
    print(f"-> ERROR FATAL: El texto extraído no es un JSON válido. Detalles: {e}")
    exit()
except Exception as e:
    print(f"-> ERROR FATAL: Ocurrió un error inesperado al procesar el JSON. Detalles: {e}")
    exit()

# --- Paso 5: Procesar los datos y guardarlos ---
print("\n[Paso 5/5] Navegando el JSON y guardando los datos...")
try:
    # La ruta para llegar a los datos históricos
    historical_data = full_data['state']['page']['gfs']

    observations = historical_data['observations']['temperature']['y']
    timestamps = historical_data['observations']['temperature']['x']

    dates = [datetime.fromtimestamp(ts) for ts in timestamps]

    df = pd.DataFrame({
        'Fecha y Hora': dates,
        'Temperatura_C': observations
    })

    output_file = 'datos_temperatura_cdmx_2023_DIAGNOSTICO.csv'
    df.to_csv(output_file, index=False)

    print(f"-> ÉXITO TOTAL: Los datos se han guardado en '{output_file}'")
    print("\n--- DIAGNÓSTICO FINALIZADO ---")
    print("\nPrimeras 5 filas del resultado:")
    print(df.head())

except KeyError as e:
    print(f"-> ERROR FATAL: La estructura del JSON ha cambiado. No se encontró la clave: {e}")
    print("   Revisa la ruta ['state']['page']['gfs']['observations']['temperature'] dentro del JSON.")
    exit()
except Exception as e:
    print(f"-> ERROR FATAL: Ocurrió un error al procesar y guardar los datos. Detalles: {e}")
    exit()
    