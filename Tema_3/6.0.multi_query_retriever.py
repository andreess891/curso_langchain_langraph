import httpx

from dotenv import load_dotenv,find_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", http_client=http_client),
    persist_directory="chroma_database/"
)

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    http_client=http_client
)

base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

consulta = "¿Donde se enceuntra el local del contrato en el que participa María Jiménez Campos?"

resultados = retriever.invoke(consulta)

print("Top documentos mas similares a la consulta: \n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido del documento {i}:\n{doc.page_content}\n")
    print(f"Metadatos del documento {i}:\n{doc.metadata}\n")