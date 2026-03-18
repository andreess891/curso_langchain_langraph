import httpx

from dotenv import load_dotenv,find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter # Solucionar el problema de texto largo dividiendolo en partes mas pequeñas

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

# 1. Cargar el documento PDF
loader = PyPDFLoader("quijote.pdf")
pages = loader.load()

# Dividir el texto en chunks mas pequeños para evitar el error de texto largo
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000, 
    chunk_overlap=200
)

# Dividir cada pagina en chunks mas pequeños con la configuracion anterior
chunks = text_splitter.split_documents(pages)


# 3. Pasar el texto al LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    http_client=http_client
)

summaries = []
for chunk in chunks:
    response = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente texto: {chunk.page_content}")
    summaries.append(response.content)

final_summary = llm.invoke(f"Combina y sintetiza estos resumenes en un resumen coherente y completo: {' '.join(summaries)}")

print(final_summary.content)