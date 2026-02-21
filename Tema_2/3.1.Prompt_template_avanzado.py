

 
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor del español al inglés muy preciso y profesional."),
    ("human", "Traduce el siguiente texto: {texto}")
])

mensajes = chat_prompt.format_messages(texto="Hola, ¿cómo estás?")

for m in mensajes:
    print(f"{type(m)}: {m.content}")