from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from operator import attrgetter

import httpx
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

# Asegurarse de que el modelo utilizado es compatible con herramientas personalizadas, como gpt-4o-mini o gpt-4o. En este ejemplo se utiliza gpt-4o-mini.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, http_client=http_client)

@tool("user_db_tool")
def herramienta_personalizada(query: str) -> str:
    """Consulta la base de usuarios de la empresa."""
    # Codigo que accede a la basede datos
    return f"Respuesta a la consulta: {query}"


llm_with_tools = llm.bind_tools([herramienta_personalizada])

chain =  llm_with_tools | attrgetter("tool_calls") | herramienta_personalizada.map()

response = chain.invoke("Generea un resumen de la informacion que hay en la base de datos para el usuario con id 12345")

print(response)


print("\n\n")

print(response[0].content)
print("\n\n")
# Si intento invocar la cadena con una consulta donde el modelo determina que no es necesario usar la herramienta personalizada, la respuesta queda vacia ya que 
# el modelo no ha decidido usar la herramienta personalizada para responder a la consulta, por lo que no hay nada en content ni en tool_calls.
response = chain.invoke("Cual es la capital de Alemania?")
print(response)
