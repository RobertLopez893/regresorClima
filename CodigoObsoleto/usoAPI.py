import requests
import pandas as pd

print("Obteniendo datos históricos del clima para la Ciudad de México a través de la API...")

# 1. Coordenadas de la Ciudad de México
latitude = 19.4326
longitude = -99.1332

# 2. Parámetros para la API
#    - Queremos la temperatura ('temperature_2m')
#    - Para el año 2023 ('start_date' y 'end_date')
#    - Con datos horarios ('hourly')
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "hourly": "temperature_2m"  # temperature_2m es la temperatura del aire a 2 metros de altura
}

# 3. URL de la API de Open-Meteo
url = "https://archive-api.open-meteo.com/v1/archive"

# 4. Hacemos la solicitud a la API
try:
    response = requests.get(url, params=params)
    response.raise_for_status()  # Lanza un error si la solicitud no fue exitosa (e.g., error 404 o 500)
    data = response.json()
    print("¡Datos recibidos exitosamente!")

except requests.exceptions.RequestException as e:
    print(f"Error al conectar con la API: {e}")
    exit()

# 5. Procesamos la respuesta y la convertimos a un DataFrame de Pandas
if 'hourly' in data:
    hourly_data = data['hourly']
    df = pd.DataFrame(data={
        "Fecha y Hora": pd.to_datetime(hourly_data['time']),
        "Temperatura_C": hourly_data['temperature_2m']
    })

    # 6. Guardamos los datos en un archivo CSV
    output_file = 'datos_temperatura_cdmx_2023_API.csv'
    df.to_csv(output_file, index=False)

    print(f"\n¡Éxito! Se han guardado {len(df)} registros en el archivo '{output_file}'")
    print("\nPrimeras 5 filas de datos:")
    print(df.head())
    print("\nÚltimas 5 filas de datos:")
    print(df.tail())

else:
    print("La respuesta de la API no contiene los datos horarios esperados.")
    print("Respuesta recibida:", data)
