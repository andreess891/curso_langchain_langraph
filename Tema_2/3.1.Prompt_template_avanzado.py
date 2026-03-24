import httpx
from dotenv import load_dotenv,find_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

http_client = httpx.Client(verify=False)

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.7,
    http_client=http_client
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor del español al inglés muy preciso y profesional."),
    ("human", "Traduce el siguiente texto: {texto}")
])

mensajes = chat_prompt.format_messages(texto="Hola, ¿cómo estás?")

for m in mensajes:
    print(f"{type(m)}: {m.content}")

cadena = chat_prompt | llm
resultado = cadena.invoke({"texto":"Hola, ¿cómo estás?"})
print(f"Respuesta del modelo: {resultado.content}")