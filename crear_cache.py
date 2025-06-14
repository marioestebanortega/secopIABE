import requests
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuración general
BASE_PROCESOS = "https://www.datos.gov.co/resource/p6dx-8zbt.json"
BASE_CONTRATOS = "https://www.datos.gov.co/resource/jbjy-vk9h.json"
CACHE_DIR = "cache_secop"
LIMIT = 1000
SLEEP_TIME = 1


PALABRAS_CLAVE = [
    "inteligencia artificial"
]

os.makedirs(CACHE_DIR, exist_ok=True)

def contiene_palabra_clave(texto):
    if not texto:
        return False
    texto = texto.lower()
    return any(palabra in texto for palabra in PALABRAS_CLAVE)

def obtener_datos_paginados(url, params=None):
    all_data = []
    offset = 0
    while True:
        params_paginados = params.copy() if params else {}
        params_paginados.update({"$limit": LIMIT, "$offset": offset})
        print(f"🔄 Descargando registros {offset + 1} a {offset + LIMIT}...")
        print("🚨 URL:", url)
        print("params_paginados",params_paginados)
        response = requests.get(url, params=params_paginados)
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < LIMIT:
            break
        offset += LIMIT
        time.sleep(SLEEP_TIME)
    return all_data



def guardar_json(nombre, datos):
    with open(os.path.join(CACHE_DIR, nombre), "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

def main():
    procesos = obtener_datos_paginados(BASE_PROCESOS, {
    #"$where": "fecha_de_publicacion_del >= '2024-01-01'",
    "$where": "upper(descripci_n_del_procedimiento) like '%INTELIGENCIA%ARTIFICIAL%' AND fecha_de_publicacion_del >= '2024-05-01'"
})
    contratos = obtener_datos_paginados(BASE_CONTRATOS, {
      #  "$where": "ultima_actualizacion >= '2024-01-01'",
       "$where": "upper(objeto_del_contrato) like '%INTELIGENCIA%ARTIFICIAL%' AND ultima_actualizacion >= '2024-05-01'"
    })

    guardar_json(f"procesos_ia_abiertos.json", procesos)
    guardar_json(f"contratos_completos.json", contratos)

    print(f"✅ Procesos filtrados: {len(procesos)}")
    print(f"✅ Contratos filtrados: {len(contratos)}")

if __name__ == "__main__":
    main()