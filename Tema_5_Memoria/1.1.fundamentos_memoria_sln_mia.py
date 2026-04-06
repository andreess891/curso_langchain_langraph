# El siguiete código muestra un ejemplo de cómo usar LangChain para crear un asistente de chat simple en la terminal. 
# El asistente responde a las entradas del usuario utilizando el modelo GPT-4o-mini de OpenAI.
# Sin embargo, el código no incluye ningún manejo de memoria o contexto a largo plazo, lo que significa que cada respuesta 
# se genera de forma independiente sin recordar las interacciones anteriores.


# La siguiente contiene la solucion del manejo de memoria que se me ocurre 
# usando MessagesPlaceholder, pero no se si es la mejor forma de hacerlo.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

import httpx
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=http_client)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

historial_conversacion = []

chain = prompt | llm

print("Chat en terminal (escribe 'salir' para terminar)\n")

while True:
    try:
        user_input = input("Tú: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nHasta luego!")
        break

    if not user_input:
        continue
    if user_input.lower() in {"salir", "exit", "quit"}:
        print("Hasta luego!")
        break

    respuesta = chain.invoke({"input": user_input, "history": historial_conversacion})
    historial_conversacion.append(HumanMessage(content=user_input))
    historial_conversacion.append(AIMessage(content=respuesta.content))
    print("Asistente:", respuesta.content)