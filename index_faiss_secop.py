from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import json
import os
import requests
import time
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
CACHE_DIR = "cache_secop"
INDICE_DIR = "indice_secop"
BATCH_SIZE = 200     # Tamaño del lote para embeddings
MAX_RETRIES = 3     # Número máximo de reintentos por lote
INITIAL_WAIT = 30   # Tiempo inicial de espera entre reintentos

# Crear directorios si no existen
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(INDICE_DIR, exist_ok=True)

def obtener_datos():
    """Obtiene datos de la caché"""
    # Procesos
    procesos_file = os.path.join(CACHE_DIR, "procesos_ia_abiertos.json")
    if not os.path.exists(procesos_file):
        raise FileNotFoundError("No se encontró la caché de procesos. Ejecuta primero crear_cache.py")
    
    print("Cargando procesos desde caché...")
    with open(procesos_file, "r", encoding="utf-8") as f:
        procesos = json.load(f)
    print(f"✅ Procesos cargados: {len(procesos)}")

    # Contratos
    contratos_file = os.path.join(CACHE_DIR, "contratos_completos.json")
    if not os.path.exists(contratos_file):
        raise FileNotFoundError("No se encontró la caché de contratos. Ejecuta primero crear_cache.py")
    
    print("Cargando contratos desde caché...")
    with open(contratos_file, "r", encoding="utf-8") as f:
        contratos = json.load(f)
    print(f"✅ Contratos cargados: {len(contratos)}")

    return procesos, contratos

# Obtener datos
procesos, contratos = obtener_datos()

# Crear documentos tipo LangChain (1 por registro JSON)
docs = []

def json_a_documento(registro, tipo):
    """Convierte un registro JSON en un documento LangChain"""
    texto = json.dumps(registro, ensure_ascii=False)
    metadata = {
        "tipo": tipo,
        "id": registro.get("id_del_proceso") or registro.get("id_contrato"),
        "entidad": registro.get("entidad", ""),
        "descripcion": registro.get("descripci_n_del_procedimiento", ""),
        "valor": registro.get("precio_base") or registro.get("valor_del_contrato", "")
    }
    return Document(page_content=texto, metadata=metadata)

print("\nCreando documentos...")
for p in procesos:
    docs.append(json_a_documento(p, "proceso"))

for c in contratos:
    docs.append(json_a_documento(c, "contrato"))

print(f"✅ Total de documentos: {len(docs)}")

# Embeddings Gemini
print("\nCreando embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Procesar documentos en lotes
all_embeddings = []
total_lotes = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
start_time = time.time()

for i in range(0, len(docs), BATCH_SIZE):
    batch = docs[i:i + BATCH_SIZE]
    lote_actual = i // BATCH_SIZE + 1
    
    print(f"\nProcesando lote {lote_actual} de {total_lotes}...")
    
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # Obtener embeddings para el lote actual
            batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch])
            all_embeddings.extend(batch_embeddings)
            
            # Esperar antes del siguiente lote
            #if i + BATCH_SIZE < len(docs):
            #    print(f"Esperando {SLEEP_TIME} segundos...")
            #    time.sleep(SLEEP_TIME)
            break
            
        except Exception as e:
            retries += 1
            if retries == MAX_RETRIES:
                print(f"Error en lote {lote_actual} después de {MAX_RETRIES} intentos: {str(e)}")
                raise
            wait_time = INITIAL_WAIT * (2 ** (retries - 1))  # Espera exponencial: 30s, 60s, 120s
            print(f"Error en lote {lote_actual} (intento {retries}/{MAX_RETRIES}): {str(e)}")
            print(f"Esperando {wait_time} segundos antes de reintentar...")
            time.sleep(wait_time)

# Crear y guardar el índice
print("\nCreando índice FAISS...")
vectorstore = FAISS.from_embeddings(
    text_embeddings=list(zip([doc.page_content for doc in docs], all_embeddings)),
    embedding=embeddings,
    metadatas=[doc.metadata for doc in docs]
)

# Guardar índice local
print("Guardando índice...")
vectorstore.save_local(INDICE_DIR)
tiempo_total = (time.time() - start_time) / 60
print(f"\n✅ Índice vectorial creado y guardado en '{INDICE_DIR}'")
print(f"⏱️ Tiempo total de procesamiento: {tiempo_total:.1f} minutos")
print(f"📊 Total de documentos indexados: {len(docs)}")
