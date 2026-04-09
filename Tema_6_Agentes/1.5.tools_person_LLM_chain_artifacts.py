from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from operator import attrgetter
from typing import Tuple

import httpx
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

# Asegurarse de que el modelo utilizado es compatible con herramientas personalizadas, como gpt-4o-mini o gpt-4o. En este ejemplo se utiliza gpt-4o-mini.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, http_client=http_client)

@tool("user_db_tool", response_format="content_and_artifact")
def herramienta_personalizada(query: str) -> Tuple[str, dict]:
    """Consulta la base de usuarios de la empresa."""
    # Codigo que accede a la basede datos
    return f"Respuesta a la consulta: {query}", 10


llm_with_tools = llm.bind_tools([herramienta_personalizada])

chain =  llm_with_tools | attrgetter("tool_calls") | herramienta_personalizada.map()

response = chain.invoke("Genera un resumen de la nformacion sobre el usuario con id 12345 que se encuentra en nuestra base de datos")

print(response)
