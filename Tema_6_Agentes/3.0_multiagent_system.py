# Exiten varios tipos de sistemas multiagente:
# - Red de agentes (Network): Varios agentes independientes que interactúan entre sí, pero no hay un control centralizado.
# - Supervisor central (Supervisor): Un agente supervisa y coordina a otros agentes, asignándoles tareas y gestionando su trabajo.
# - Equipos jerarquicos (Hierarchical): Agentes organizados en niveles, donde los agentes de nivel superior supervisan a los de nivel inferior.

# En este ejemplo, implementaremos un sistema multiagente con un supervisor central que coordina a varios agentes especializados en diferentes tareas.

from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import httpx
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=http_client)

# Definir herramientas personalizadas
@tool
def buscar_web(query: str) -> str:
    """Buscar informacion en la web."""
    return f"Resultados de búsqueda para: {query}"

@tool
def calcular(expression: str) -> str:
    """Realizar cálculos matemáticos."""
    return f"Resultado del cálculo: {eval(expression)}"

# Crear agentes especializados
agente_investigacion = create_react_agent(
    model=model,
    tools=[buscar_web],
    prompt="Eres un especializata en invetigacion web.",
    name="investigador"
)

agente_matematicas = create_react_agent(
    model=model,
    tools=[calcular],
    prompt="Eres un especilista en calculos matematicos.",
    name="matematico"
)

# Crear supervisor que coordina los agente
supervisor_graph = create_supervisor(
    [agente_investigacion, agente_matematicas],
    model=model,
    prompt="Eres un supervisor que delega tareas a especialistas segun el tipo de consulta."
)

supervisor = supervisor_graph.compile()

# Uso del sistema multiagente
response = supervisor.invoke({
    "messages": [{
        "role": "user",
        "content": "Busca informacion sobre pi y calcula su valor multiplicado por 2."
    }]
})

for msg in response['messages']:
    print(msg.content)