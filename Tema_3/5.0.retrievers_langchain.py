import httpx
import os

from dotenv import load_dotenv,find_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

# Los retrievers son componentes que permiten recuperar información relevante de un vector store basado en una consulta. 
# En este ejemplo, se utiliza el método as_retriever de Chroma para crear un retriever que realiza búsquedas de similitud 
# en el vector store, devolviendo los documentos más relevantes para una consulta dada. Esto facilita la recuperación de 
# información específica dentro de un conjunto de documentos almacenados

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", http_client=http_client),
    persist_directory="chroma_database/"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

consulta = "¿Donde se enceuntra el local del contrato en el que participa María Jiménez Campos?"

resultados = retriever.invoke(consulta)

print("Top 2 documentos mas similares a la consulta: \n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido del documento {i}:\n{doc.page_content}\n")
    print(f"Metadatos del documento {i}:\n{doc.metadata}\n")