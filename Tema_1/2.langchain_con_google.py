from dotenv import load_dotenv,find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7
)

pregunta = "¿Cuál es la capital de Francia?. Solo dame la respuesta sin explicaciones."
print(f"pregunta: {pregunta}")

respuesta = llm.invoke(pregunta)

print(f"Respuesta del modelo: {respuesta.content}")