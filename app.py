import requests
import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Configuración
BASE_PROCESOS = "https://www.datos.gov.co/resource/p6dx-8zbt.json"
BASE_CONTRATOS = "https://www.datos.gov.co/resource/jbjy-vk9h.json"
CACHE_DIR = "cache_secop"
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Cargar el índice vectorial
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
try:
    vectorstore = FAISS.load_local("indice_secop", embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("✅ Índice vectorial cargado correctamente")
except Exception as e:
    print("⚠️ No se encontró el índice vectorial. Ejecuta primero index_faiss_secop.py para crearlo.")
    print(f"Error: {str(e)}")
    vectorstore = None
    retriever = None

# LLM Gemini Pro
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    convert_system_message_to_human=True
)

# Prompt personalizado
template = """Eres un asesor experto en contratación pública colombiana. Con base en los siguientes documentos del SECOP II:

{context}

Responde en lenguaje claro:
1. ¿Qué contratos o procesos son relevantes para la pregunta?
2. ¿Qué entidades están involucradas?
3. ¿Cuál es el presupuesto o valor?

Incluye los IDs relevantes.

Pregunta: {question}
"""

prompt = PromptTemplate.from_template(template)

qa_chain = None
if retriever:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )

@app.route("/procesos", methods=["POST"])
def buscar_procesos():
    data = request.get_json()
    palabra_clave = data.get("palabra_clave", "")
    if not palabra_clave:
        return jsonify({"error": "Falta el campo 'palabra_clave'"}), 400

    procesos_file = os.path.join(CACHE_DIR, "procesos_ia_abiertos.json")
    contratos_file = os.path.join(CACHE_DIR, "contratos_completos.json")

    # Procesos: usa caché general si existe
    if os.path.exists(procesos_file):
        with open(procesos_file, "r", encoding="utf-8") as f:
            procesos_abiertos = json.load(f)
    else:
        print("Obteniendo todos los procesos abiertos...")
        where_clause = "estado_de_apertura_del_proceso = 'Abierto'"
        params = {"$where": where_clause}
        procesos_abiertos = requests.get(BASE_PROCESOS, params=params).json()
        os.makedirs(os.path.dirname(procesos_file), exist_ok=True)
        with open(procesos_file, "w", encoding="utf-8") as f:
            json.dump(procesos_abiertos, f, ensure_ascii=False, indent=2)
        print(f"✅ Procesos obtenidos: {len(procesos_abiertos)}")

    # Filtrar procesos por palabra clave en memoria
    palabra_clave_upper = palabra_clave.upper()
    procesos = []
    
    # Convertir cada proceso a string JSON y buscar la palabra clave
    for p in procesos_abiertos:
        # Convertir el proceso a string JSON
        proceso_str = json.dumps(p, ensure_ascii=False)
        if palabra_clave_upper in proceso_str.upper():
            procesos.append(p)

    # Contratos: usa caché si existe
    if os.path.exists(contratos_file):
        with open(contratos_file, "r", encoding="utf-8") as f:
            todos_los_contratos = json.load(f)
    else:
        print("Obteniendo todos los contratos...")
        params = {}
        todos_los_contratos = requests.get(BASE_CONTRATOS, params=params).json()
        with open(contratos_file, "w", encoding="utf-8") as f:
            json.dump(todos_los_contratos, f, ensure_ascii=False, indent=2)
        print(f"✅ Contratos obtenidos: {len(todos_los_contratos)}")

    return jsonify({"procesos": procesos})

@app.route("/preguntar", methods=["POST"])
def preguntar():
    if not qa_chain:
        return jsonify({
            "error": "El índice vectorial no está disponible. Ejecuta primero index_faiss_secop.py para crearlo."
        }), 503
    
    data = request.get_json()
    pregunta = data.get("pregunta")
    if not pregunta:
        return jsonify({"error": "Falta el campo 'pregunta'"}), 400
    
    # Usar invoke() en lugar de run() para manejar múltiples outputs
    result = qa_chain.invoke({"query": pregunta})
    respuesta = result["result"]
    documentos = result["source_documents"]
    
    # Procesar documentos para agregar campos necesarios
    documentos_procesados = []
    for doc in documentos:
        try:
            contenido = doc.page_content
            if isinstance(contenido, str):
                try:
                    # Intentar parsear como JSON
                    doc_dict = json.loads(contenido)
                    
                    # Agregar campos necesarios si no existen
                    if "id_adjudicacion" not in doc_dict:
                        doc_dict["id_adjudicacion"] = doc_dict.get("id_contrato", "No Adjudicado")
                    
                    if "precio_base" not in doc_dict:
                        doc_dict["precio_base"] = doc_dict.get("valor_del_contrato", "No especificado")
                    
                    # Mantener urlproceso como está si existe, o crear uno vacío si no
                    if "urlproceso" not in doc_dict:
                        doc_dict["urlproceso"] = doc.metadata.get("url", "") if isinstance(doc.metadata.get("url", ""), str) else ""
                    
                
                    # Mantener urlproceso como está si existe, o crear uno vacío si no
                    if "urlproceso" not in doc_dict:
                        doc_dict["urlproceso"] = doc.metadata.get("url", "")
                    if isinstance(doc_dict["urlproceso"], dict):
                        doc_dict["urlproceso"] = doc_dict["urlproceso"].get("url", "")
                    
                    # Agregar estado_resumen si no existe
                    if "estado_resumen" not in doc_dict:
                        if "proveedor_adjudicado" in doc_dict:
                            doc_dict["estado_resumen"] = "Adjudicado"
                        else:
                            doc_dict["estado_resumen"] = "Presentación de oferta"
                    
                    # Asegurar que el documento termine correctamente
                    if not contenido.strip().endswith("}"):
                        contenido = contenido.rstrip() + "}"
                    
                    documentos_procesados.append(json.dumps(doc_dict, ensure_ascii=False))
                except json.JSONDecodeError:
                    # Si no es JSON válido, buscar en el texto
                    if not contenido.strip().endswith("}"):
                        contenido = contenido.rstrip() + "}"
                    
                    if "proveedor_adjudicado" in contenido:
                        contenido = contenido.replace("}", ', "estado_resumen": "Adjudicado", "id_adjudicacion": "No Adjudicado", "urlproceso": ""}')
                    else:
                        contenido = contenido.replace("}", ', "estado_resumen": "Presentación de oferta", "id_adjudicacion": "No Adjudicado", "urlproceso": ""}')
                    documentos_procesados.append(contenido)
        except Exception as e:
            print(f"Error procesando documento: {str(e)}")
            documentos_procesados.append(str(doc))
    
    return jsonify({
        "pregunta": pregunta,
        "respuesta": respuesta,
        "documentos_fuente": documentos_procesados
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
