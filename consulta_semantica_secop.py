from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Cargar el índice
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("indice_secop", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Modelo Gemini Pro
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.2,
    convert_system_message_to_human=True
)

# Prompt para análisis
template = PromptTemplate.from_template("""
Eres un asesor experto en contratación pública colombiana. Con base en los siguientes documentos recuperados del SECOP II:

{text}

Responde de forma clara y en lenguaje natural:
1. ¿Qué contratos o procesos son relevantes para la pregunta del usuario?
2. ¿Qué entidades están involucradas?
3. ¿Cuál es el presupuesto o valor de los contratos?

Incluye los identificadores de los procesos o contratos relevantes al final.
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": template}
)

# Pregunta de ejemplo
task = "¿Qué oportunidades hay en inteligencia artificial aplicadas a educación o salud?"
respuesta = qa_chain.run(task)
print("🔎 Pregunta:", task)
print("\n🧠 Respuesta:", respuesta)
