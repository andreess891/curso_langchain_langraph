from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Los MessagesPlaceholder permiten incluir una lista de mensajes en el prompt, manteniendo el contexto de la conversación. En este ejemplo, se utiliza para incluir el historial de la conversación antes de la pregunta actual del usuario.

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "{pregunta_actual}"),
])

historial_conversacion = [
    HumanMessage(content="¿Cuál es la capital de Francia?"),
    AIMessage(content="La capital de Francia es París."),
    HumanMessage(content="¿Y cuántos habitantes tiene?"),
    AIMessage(content="París tiene aproximadamente 2.2 millones de habitantes en la ciudad propiamente dicha.")
]

mensajes = chat_prompt.format_messages(
    historial = historial_conversacion,
    pregunta_actual = "¿Puedes decirme algo interesante de su arquitectura?"
)

for mensaje in mensajes:
    print(f"{mensaje.type}: {mensaje.content}")