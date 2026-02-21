# Código que muestra como usar PromptTemplate y como probarlo antes de enviarlo al LLM con la funcion format. 
# Esto es útil para asegurarnos de que el prompt se construye correctamente antes de enviarlo al modelo de lenguaje.
 
from langchain_core.prompts import PromptTemplate

template = "Eres un experto en marketing. Sugiere un eslogan para un producto que es: {producto}."

prompt = PromptTemplate(
    template=template,
    input_variables=["producto"]
)

prompt_completado = prompt.format(producto="una bebida energética saludable")

print(prompt_completado)