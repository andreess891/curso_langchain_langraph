from langgraph.graph import MessagesState, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import chromadb
from langchain_chroma import Chroma
import uuid

import httpx
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

CHROMA_PATH = "./Tema_5_Memoria/chromadb"

# Configuracion basica del LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=http_client)

# Configuracion de la base de datos vectorial con ChromaDB
vectorstore = Chroma(
    collection_name="memoria_chat", # Nombre de la colección para almacenar las memorias del chat
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", http_client=http_client), # Función de embedding para convertir texto a vectores
    persist_directory=CHROMA_PATH # Directorio donde se almacenarán los datos de ChromaDB
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("memoria_chat")

def guardar_memoria(texto):
    """Guarda informacon relevante del usuario en la base de datos vectorial"""
    try:
        collection.add(
            documents=[texto],
            ids=[str(uuid.uuid4())]
        )
        print(f"[+] Guardado en memoria vectorial: {texto}")
    except Exception as e:
        print(f"Error al guardar en memoria vectorial: {e}")

def buscar_memoria(consulta, k=3):
    """Busca informacion relevante en la memoria vectorial de chromadb por similitud semantica"""
    try:
        results = collection.query(
            query_texts=[consulta],
            n_results=k
        )
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"Error al buscar en memoria vectorial: {e}")
        return []
    
def chatbot_node(state):
    """Nodo principal del grafo."""
    messages = state["messages"]
    ultimo_mensaje = messages[-1].content if messages else ""

    # 1. Buscar memorias relevantes
    memorias = buscar_memoria(ultimo_mensaje)

    # 2. Crear prompt con memorias
    system_content = "Eres un asistente que recuerda informacion importante."
    if memorias:
        system_content += "\n\nInformacion que recuerdas:"
        for memoria in memorias:
            system_content += f"\n- {memoria}"

    # 3. Generar respuesta
    messages_con_sistema = [SystemMessage(content=system_content)] + messages
    response = llm.invoke(messages_con_sistema)

    # 4. Guardar informacion relevante del usuario en la memoria vectorial
    mensaje_usuario_lower = ultimo_mensaje.lower()
    if "me llamo" in mensaje_usuario_lower or "mi nombre es" in mensaje_usuario_lower:
        guardar_memoria(f"el usuario se llama: {ultimo_mensaje}")
    elif any(frase in mensaje_usuario_lower for frase in ["trabajo en", "trabajo como", "Soy programador", "Soy ingeniero"]):
        guardar_memoria(f"Trabajo del usuario: {ultimo_mensaje}")
    elif "me gusta" in mensaje_usuario_lower or "me encanta" in mensaje_usuario_lower:
        guardar_memoria(f"Gustos del usuario: {ultimo_mensaje}")
    elif "vivo en" in mensaje_usuario_lower or "soy de" in mensaje_usuario_lower:
        guardar_memoria(f"Ubicacion del usuario: {ultimo_mensaje}")

    return {"messages": [response]}

# Crear el grafo
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")

# Compilar con memoria volatil de conversacion
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def chat(message, thread_id="default_thread"):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": HumanMessage(content=message)}, config=config)
    return result["messages"][-1].content

def mostrar_memorias():
    """Funcion auxiliar para ver todas las memorias guardadas del usuario en la base de datos vectorial"""
    try:
        all_memorias = collection.get()
        if all_memorias['documents']:
            print("[+] Memorias guardadas en la base de datos vectorial:")
            for i, memoria in enumerate(all_memorias['documents'], 1):
                print(f"{i}. {memoria}")
        else:
            print("[-] No hay memorias guardadas aun en la base de datos vectorial.")
    except Exception as e:
        print(f"Error obteniendo memorias: {e}")

if __name__ == "__main__":
    print("Chat en terminal (escribe 'salir' para terminar, 'memorias' para ver memoria historica)\n")
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
            
        if user_input.lower() == "memorias":
            mostrar_memorias()
            continue
        
        # Ejecutar la cadena con memoria para obtener la respuesta del asistente
        respuesta = chat(user_input, session_id)

        print("Asistente:", respuesta)
        print()