#Este código es una forma de utilizar LangChain usando plantilas de prompts y cadenas (chains) para estructurar la interacción con el modelo de lenguaje.
#Este código está obsoleto, se recomienda usar el código "4.langchain_con_openai_avanzado_LCEL.py" ya que utiliza la nueva sintaxis de LangChain Expression Language (LCEL) para lograr lo mismo de una forma mas sencilla y elegante.

import httpx

from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

chain = LLMChain(
    llm=chat, 
    prompt=plantilla
)

resultado =chain.run(nombre="Jonatan", pais="Francia")

print(f"Respuesta del modelo: {resultado}")