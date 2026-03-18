import httpx
import os

from dotenv import load_dotenv,find_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

loader = PyPDFDirectoryLoader("contratos")
documentos = loader.load()

print(f"se cargaron {len(documentos)} documentos desde el directorio")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000, 
    chunk_overlap=1000
)

docs_split = text_splitter.split_documents(documentos)

print(f"se generaron {len(docs_split)} fragmentos de texto después de la división")

vector_store = Chroma.from_documents(
    documents=docs_split,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large", http_client=http_client),
    persist_directory="chroma_database"
)

consulta = "¿Donde se enceuntra el local del contrato en el que participa María Jiménez Campos?"

resultados = vector_store.similarity_search(consulta, k=2)

print("Top 2 documentos mas similares a la consulta:")
for i, doc in enumerate(resultados, 1):
    print(f"Contenido del documento {i}:\n{doc.page_content}\n")
    print(f"Metadatos del documento {i}:\n{doc.metadata}\n")