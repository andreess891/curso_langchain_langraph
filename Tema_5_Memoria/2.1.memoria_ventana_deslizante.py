# En este ejemplo se muestra cómo manejar la memoria de tipo ventana deslizante en un chat usando LangGraph, 
# lo que permite mantener el contexto de la conversación a lo largo de múltiples interacciones, 
# pero limitando la cantidad de mensajes almacenados para evitar problemas de memoria.
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI

# Gestion de la memoria con LangGraph
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

import httpx
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=http_client)

# Es una clase dummy para hacer referencia de que la tecnica de memoria que se va a usar sera el de ventana deslizante, al final sigue siendo lo mismo
class WindowedState(MessagesState):
    pass

workflow = StateGraph(state_schema=WindowedState)

trimmer = trim_messages(
    strategy="last", # Mantener solo los últimos mensajes
    max_tokens=4,# Limitar a los 4 ultimos mensajes (ajustable según tus necesidades)
    token_counter=len,# Definir como se cuentan los tokens, para este caso un token corresponde a un mensaje completo.
    start_on="human", # Asegurar que siempre se mantenga el último mensaje del usuario, incluso si se supera el límite de tokens.
    include_system=True # Incluir el mensaje del sistema en el recorte para mantener el contexto general de la conversación.
)

def chatbot_node(state):
    """Nodo que procesa mensajes y genera respuestas"""
    trimmed_messages = trimmer.invoke(state["messages"]) # Aplicar la función de recorte a los mensajes almacenados en el estado
    system_prompt = "Eres un asistente amigable que recuerda conversaciones previas."
    messages = [SystemMessage(content=system_prompt)] + trimmed_messages # Combinar el mensaje del sistema con los mensajes recortados para mantener el contexto necesario para la respuesta del modelo.
    response = llm.invoke(messages)

    return {"messages": [response]}

# Agregar los nodos al grafo
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")

# Compilar el grafo
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def chat(message, thread_id="default_thread"):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    return result["messages"][-1].content

if __name__ == "__main__":
    print("Chat en terminal (escribe 'salir' para terminar)\n")
    session_id = "default_thread"  # En un entorno real, esto podría ser un ID de usuario o sesión único

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
        respuesta = chat(user_input, session_id)

        print("Asistente:", respuesta)