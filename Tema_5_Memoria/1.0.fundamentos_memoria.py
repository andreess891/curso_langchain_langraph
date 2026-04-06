# El siguiete código muestra un ejemplo de cómo usar LangChain para crear un asistente de chat simple en la terminal. 
# El asistente responde a las entradas del usuario utilizando el modelo GPT-4o-mini de OpenAI.
# Sin embargo, el código no incluye ningún manejo de memoria o contexto a largo plazo, lo que significa que cada respuesta 
# se genera de forma independiente sin recordar las interacciones anteriores.

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil."),
    ("human", "{input}")
])

chain = prompt | llm

print("Chat en terminal (escribe 'salir' para terminar)\n")

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

    respuesta = chain.invoke({"input": user_input})
    print("Asistente:", respuesta.content)