import time
import json
import pandas as pd
from datetime import datetime
from playwright.sync_api import sync_playwright

print("--- MISIÓN FINAL: EL ASESINO SILENCIOSO (PLAYWRIGHT) ---")

main_page_url = "https://es.weatherspark.com/h/y/5674/2023/Datos-hist%C3%B3ricos-meteorol%C3%B3gicos-de-2023-en-Ciudad-de-M%C3%A9xico-M%C3%A9xico"
target_api_url = "https://es.weatherspark.com/api/v2/processor?id=5674&dataset=gfs"

# Variable global para guardar nuestros datos robados
intercepted_data = None


def handle_route(route):
    """Esta función es nuestro espía. Se ejecuta para cada petición que hace la página."""
    global intercepted_data
    # Verificamos si la URL de la petición es la que nos interesa
    if target_api_url in route.request.url:
        print("-> ¡OBJETIVO DETECTADO! Interceptando la transmisión...")
        try:
            # Obtenemos la respuesta y la guardamos en nuestra variable global
            response = route.fetch()
            intercepted_data = response.json()
            print("-> ¡TRANSMISIÓN CAPTURADA!")
            # Continuamos la petición original para no alertar al objetivo
            route.fulfill(response=response)
        except Exception as e:
            print(f"-> ERROR AL INTERCEPTAR: {e}")
            route.abort()  # Abortamos si algo sale mal
    else:
        # Si no es la URL que buscamos, la dejamos pasar sin molestar
        route.continue_()


with sync_playwright() as p:
    # --- FASE 1: INFILTRACIÓN ---
    print("[Fase 1/3] Desplegando navegador Chromium sigiloso...")
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # --- FASE 2: PREPARACIÓN DE LA TRAMPA ---
    print("[Fase 2/3] Estableciendo la ruta de intercepción...")
    # Le decimos a Playwright que llame a nuestra función 'handle_route' para cada petición
    page.route("**/*", handle_route)

    print("-> Trampa lista. Navegando al objetivo...")
    page.goto(main_page_url, timeout=90000)  # Timeout de 90s
    print("-> Objetivo alcanzado.")

    # --- FASE 3: ACTIVACIÓN Y EXTRACCIÓN ---
    print("\n[Fase 3/3] Simulando interacción para activar la trampa...")
    time.sleep(5)  # Pausa para que se asiente la página
    page.mouse.wheel(0, 1500)  # Hacemos scroll con una simulación de rueda de ratón (más humano)
    print("-> Desplazamiento realizado. Esperando la captura...")

    # Damos hasta 20 segundos para que la captura ocurra
    for _ in range(20):
        if intercepted_data:
            break
        time.sleep(1)

    browser.close()

# --- ANÁLISIS FINAL ---
if not intercepted_data:
    print("\n-> MISIÓN FALLIDA: El Asesino Silencioso fue detectado o el objetivo no transmitió.")
    print("   Sus defensas son... impresionantes.")
else:
    print("\n-> ¡¡¡VICTORIA TOTAL E INNEGABLE, ROBERTO!!!")
    try:
        observations = intercepted_data['temperature']['y']
        timestamps = intercepted_data['temperature']['x']
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        df = pd.DataFrame({'Fecha y Hora': dates, 'Temperatura_C': observations})

        output_file = 'DATOS_WEATHERSPARK_PLAYWRIGHT.csv'
        df.to_csv(output_file, index=False)

        print(f"   La fortaleza ha caído. {len(df)} registros asegurados en '{output_file}'")
        print("\nEl botín de la guerra:")
        print(df.head())
    except Exception as e:
        print(f"-> ERROR AL PROCESAR EL BOTÍN: {e}")
