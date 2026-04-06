# El siguiete código muestra un ejemplo de cómo usar LangChain para crear un asistente de chat simple en la terminal. 
# El asistente responde a las entradas del usuario utilizando el modelo GPT-4o-mini de OpenAI.
# Sin embargo, el código no incluye ningún manejo de memoria o contexto a largo plazo, lo que significa que cada respuesta 
# se genera de forma independiente sin recordar las interacciones anteriores.


# La siguiente contiene la solucion del manejo de memoria usando LangChain, pero es recomendable usarlo
# solo en soluciones basicas.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Para evitar problemas de memoria, se puede usar un almacenamiento en memoria para cada sesión de chat.
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

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

chain = prompt | llm

# Almacenamiento en memoria para cada sesión de chat
store = {}

# Función para obtener el historial de mensajes de una sesión específica
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Cadena automática con memoria por sesión
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

print("Chat en terminal (escribe 'salir' para terminar)\n")
session_id = "default_session"  # En un entorno real, esto podría ser un ID de usuario o sesión único

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
    
    # Ejecutar la cadena con memoria para obtener la respuesta del asistente
    respuesta = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    print("Asistente:", respuesta.content)