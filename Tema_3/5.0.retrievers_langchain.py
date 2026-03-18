import httpx
import os

from dotenv import load_dotenv,find_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

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