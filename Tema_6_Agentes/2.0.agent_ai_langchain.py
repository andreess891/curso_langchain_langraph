from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import GmailToolkit
import os

import httpx
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)


# Configurar el directorio de trabajo
#original_dir = os.getcwd()
#os.chdir("Tema_6_Agentes")

# Configurar el toolkit de Gmail
gmail_toolkit = GmailToolkit()
tools = gmail_toolkit.get_tools()

#print("Herramientas disponibles:")
#for tool in tools:
#    print(f" - {tool.name}: {tool.description}")

# Configurar modelo del agente que soporte tool calling
modelo = init_chat_model("openai:gpt-4o", temperature=0, http_client=http_client)

# Prompt de agente que definie el comportamiento
prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente de email profesional. Para procesar emails sigue EXACTAMENTE estos pasos:

    1. PRIMERO: Usa 'search_gmail' con query 'in:inbox is:unread' para obtener la lista de mensajes en la bandeja de entrada.
    
    2. SEGUNDO: De la lista obtenida, identifica el message_id del email más reciente (el primer resultado).
    
    3. TERCERO: Usa 'get_gmail_message' con el message_id real obtenido en el paso anterior para obtener el contenido completo.
    
    4. CUARTO: Analiza el email y EXTRAE esta información crítica:
       - Thread ID (busca "Thread ID:" en el contenido)
       - Remitente original (busca "From:" y extrae el email)
       - Asunto original (busca "Subject:")
       - Contenido principal del mensaje
    
    5. QUINTO: Genera una respuesta profesional y apropiada en español.
    
    6. SEXTO: Usa 'create_gmail_draft' para crear un borrador de RESPUESTA (no email nuevo) con:
       - "message": tu respuesta generada
       - "subject": "Re: [asunto original]" (si no empieza ya con "Re:")
       - "to": email del remitente original
       - "thread_id": el Thread ID extraído del paso 4 (MUY IMPORTANTE para que sea una respuesta)

    CRÍTICO PARA RESPUESTAS:
    - SIEMPRE incluye "thread_id" en create_gmail_draft para que sea una respuesta, no un email nuevo
    - El "to" debe ser el email del remitente original
    - El "subject" debe empezar con "Re:" si no lo tiene ya

    IMPORTANTE: 
    - NUNCA uses message_id hardcodeados como '1' o '2' 
    - SIEMPRE obtén los IDs reales de los mensajes primero
    - Sin thread_id, el borrador será un email nuevo, no una respuesta
    - Si no encuentras thread_id, informa el problema pero intenta crear el borrador igual
    
    Si encuentras errores, explica qué información falta y por qué."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Crear el agente con tool calling
agent = create_tool_calling_agent(modelo, tools, prompt)

# Crear executor del agente
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10 # Limitar iteraciones para evitar loops infinitos en caso de errores
)

def process_lastest_email():
    try:
        response = agent_executor.invoke({
            "input": "Procesa el email más reciente en la bandeja de entrada y genera un borrador de respuesta profesional."
        })
        return response['output']
    except Exception as e:
        print(f"Error al procesar el email: {str(e)}")
        return f"Error: {str(e)}"   
    
# Ejecutar la función para procesar el email más reciente
if __name__ == "__main__":
    result = process_lastest_email()
    print("\n" + "="*50)
    print("RESULTADO FINAL DEL AGENTE:")
    print("="*50)
    print(result)