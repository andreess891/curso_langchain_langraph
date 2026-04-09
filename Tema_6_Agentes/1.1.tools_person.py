from langchain_core.tools import tool

# Definimos una herramienta personalizada utilizando el decorador @tool
# return_direct=True indica que el resultado de la función se devuelve directamente sin procesamiento adicional, para entender mejor return_direct, es importante entender que en 
# algunos casos, las herramientas pueden devolver resultados que necesitan ser procesados o formateados antes de ser utilizados por el agente. 
# Al establecer return_direct=True, estamos indicando que el resultado de la función se devuelve tal cual, sin ningún procesamiento adicional. 
# Esto puede ser útil cuando queremos que el agente reciba la respuesta directamente y la utilice sin modificaciones.
@tool("Herramienta acceso base de datos usuarios.", return_direct=True)
def herramienta_personalizada(query: str) -> str:
    """Consulta la base de usuarios de la empresa."""
    # Codigo que accede a la basede datos
    return f"Respuesta a la consulta: {query}"


output = herramienta_personalizada.run("Consulta de prueba")
print(output)
print(f"Nombre de la herramienta: {herramienta_personalizada.name}")
print(f"Descripcion de la herramienta: {herramienta_personalizada.description}")