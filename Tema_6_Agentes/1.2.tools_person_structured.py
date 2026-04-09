# Este mecanismo casi no lo usa el profesor del curso, prediere el decorador @tool, pero es importante conocerlo para entender la evolución de las herramientas personalizadas en LangChain.
from langchain_core.tools import StructuredTool


def herramienta_personalizada2(query: str) -> str:
    """Consulta la base de usuarios de la empresa."""
    # Codigo que accede a la basede datos
    return f"Respuesta a la consulta: {query}"


mi_tool = StructuredTool.from_function(herramienta_personalizada2)

print(mi_tool.run("Consulta de prueba"))
print(f"Nombre de la herramienta: {mi_tool.name}")
print(f"Descripcion de la herramienta: {mi_tool.description}")