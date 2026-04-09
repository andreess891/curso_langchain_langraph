from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

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

response = llm_with_tools.invoke("Generea un resumen de la informacion que hay en la base de datos para el usuario con id 12345")

# Notar que en content del response no hay nada. Esto se debe a que el modelo ha decidido usar la herramienta personalizada para responder a la consulta, 
# por lo que la respuesta se encuentra en tool_calls[0] y no en content.
print(response)
print("\n\n")

# La respuesta del modelo se encuentra en tool_calls[0] ya que el modelo ha decidido usar la herramienta personalizada para responder a la consulta, por lo que no hay nada en content.
print(response.tool_calls[0])
print("\n\n")
# Se invoca la herramienta personalizada con los argumentos que el modelo ha decidido usar para responder a la consulta, y se imprime la respuesta de la herramienta personalizada.
tool_response = herramienta_personalizada.invoke(response.tool_calls[0]["args"])
print(tool_response)

