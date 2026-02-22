import httpx

from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.7,
    #api_key="sk-aaa",
    http_client=http_client
)

pregunta = "¿Cuál es la capital de Francia?. Solo dame la respuesta sin explicaciones."
print(f"pregunta: {pregunta}")

respuesta = llm.invoke(pregunta)

print(f"Respuesta del modelo: {respuesta.content}")