#Uso de LangChain Expression Language (LCEL) con OpenAI
import httpx

from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())

http_client = httpx.Client(verify=False)

chat = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.7,
    http_client=http_client
)

plantilla = PromptTemplate(
    input_variables=["pais","nombre"],
    template="Saluda al usuario con su nombre. \nNombre del usuario {nombre}\n¿Cuál es la capital de {pais}? Solo dame la respuesta sin explicaciones."
)

#Aqui está la magia de LEL, podemos encadenar la plantilla con el modelo de lenguaje de una forma muy sencilla y elegante, sin necesidad de crear una clase LLMChain ni nada por el estilo, simplemente usando el operador "|" para encadenar la plantilla con el modelo de lenguaje.
chain = plantilla | chat

resultado =chain.invoke({"nombre":"Jonatan", "pais":"Francia"})

print(f"Respuesta del modelo: {resultado.content}")