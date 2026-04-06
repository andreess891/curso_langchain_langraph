# En este ejemplo se muestra cómo manejar la memoria de un chat usando LangGraph, 
# lo que permite mantener el contexto de la conversación a lo largo de múltiples interacciones.
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Gestion de la memoria con LangGraph
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

import httpx
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=http_client)

# Definir el esquema de estado para almacenar el historial de mensajes usando MessagesState, 
# que es una estructura de datos diseñada para manejar conversaciones.
workflow = StateGraph(state_schema=MessagesState)

def chatbot_node(state):
    """Nodo que procesa mensajes y genera respuestas"""
    system_prompt = "Eres un asistente amigable que recuerda conversaciones previas."
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
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