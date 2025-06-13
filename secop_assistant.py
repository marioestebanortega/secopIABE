import os
import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from docx import Document
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SecopAssistant:
    def __init__(self, api_key: str = None, doc_path: str = None):
        """Inicializa el asistente de SECOP II.
        
        Args:
            api_key: Clave de API de Google Gemini (opcional si está en .env)
            doc_path: Ruta al documento .docx con la documentación (opcional)
        """
        # Cargar variables de entorno
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Se requiere una clave de API de Google Gemini")
            
        os.environ["GOOGLE_API_KEY"] = self.api_key
        
        # Inicializar modelo y embeddings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.2,
            convert_system_message_to_human=True
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        # Cargar documento si se proporciona
        if doc_path:
            self.load_document(doc_path)
        else:
            self.vector_store = None
            self.qa_chain = None
            self.analisis_chain = None

    def load_document(self, doc_path: str) -> None:
        """Carga y procesa el documento de documentación.
        
        Args:
            doc_path: Ruta al archivo .docx
        """
        logging.info(f"📄 Cargando documento: {doc_path}")
        
        # Cargar documento
        doc = Document(doc_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        
        # Dividir en chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text)
        
        # Crear vector store
        self.vector_store = Chroma.from_texts(
            chunks,
            self.embeddings,
            persist_directory="secop_db"
        )
        
        # Configurar memoria para conversación
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Crear chain de QA
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            return_source_documents=True
        )
        
        # Crear chain para análisis de oportunidades
        template = """
        Eres un asesor experto en contratación pública. Analiza el siguiente texto de una oportunidad en SECOP II y responde:
        1. ¿Es relevante para proyectos de inteligencia artificial?
        2. ¿Qué riesgos ves?
        3. ¿Qué recursos se necesitarían?
        4. ¿Recomendarías participar? ¿Por qué?

        Texto:
        {text}
        """
        
        prompt = PromptTemplate.from_template(template)
        self.analisis_chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        
        logging.info("✅ Documento cargado y procesado correctamente")

    def consultar(self, pregunta: str) -> Dict[str, Any]:
        """Realiza una consulta al asistente.
        
        Args:
            pregunta: Pregunta del usuario
            
        Returns:
            Dict con la respuesta y documentos fuente
        """
        if not self.qa_chain:
            return {"error": "No se ha cargado ningún documento"}
            
        try:
            result = self.qa_chain({"question": pregunta})
            return {
                "respuesta": result["answer"],
                "fuentes": [doc.page_content for doc in result["source_documents"]]
            }
        except Exception as e:
            logging.error(f"Error en consulta: {str(e)}")
            return {"error": str(e)}

    def analizar_oportunidad(self, texto: str) -> Dict[str, Any]:
        """Analiza una oportunidad de contratación.
        
        Args:
            texto: Descripción de la oportunidad
            
        Returns:
            Dict con el análisis
        """
        if not self.analisis_chain:
            return {"error": "No se ha cargado ningún documento"}
            
        try:
            result = self.analisis_chain.run(text=texto)
            return {"analisis": result}
        except Exception as e:
            logging.error(f"Error en análisis: {str(e)}")
            return {"error": str(e)}

def main():
    # Ejemplo de uso
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("❌ No se encontró la clave de API de Google")
        return
        
    assistant = SecopAssistant(api_key, "documento_secop2.docx")
    
    # Ejemplo de consulta
    pregunta = "¿Cómo puedo buscar licitaciones abiertas relacionadas con inteligencia artificial?"
    respuesta = assistant.consultar(pregunta)
    print("\n🔍 Pregunta:", pregunta)
    print("🧠 Respuesta:", respuesta["respuesta"])
    
    # Ejemplo de análisis
    oportunidad = """
    El presente proceso busca contratar el desarrollo de una solución basada en algoritmos 
    de predicción para la vigilancia epidemiológica, incluyendo procesamiento de lenguaje 
    natural y análisis de datos masivos.
    """
    analisis = assistant.analizar_oportunidad(oportunidad)
    print("\n📊 Análisis de oportunidad:")
    print(analisis["analisis"])

if __name__ == "__main__":
    main() 