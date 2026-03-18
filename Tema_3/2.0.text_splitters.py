import httpx

from dotenv import load_dotenv,find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

# 1. Cargar el documento PDF
loader = PyPDFLoader("quijote.pdf")
pages = loader.load()

# 2. Combinar todas las paginas en un texto unico
full_text = ""
for page in pages:
    full_text += page.page_content + "\n"

# 3. Pasar el texto al LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    http_client=http_client
)

# Esta peticion geneará un error porque el texto es demasiado largo para ser procesado por el modelo.
response = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente documento: {full_text}")

print(response.content)