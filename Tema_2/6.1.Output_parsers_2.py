import httpx

from dotenv import load_dotenv,find_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

class AnalisisTexto(BaseModel):
    resumen: str = Field(description="Resumen breve del texto.")
    sentimiento: str = Field(description="Sentimiento del texto (positivo, neutro o negativo)")


llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.6,
    http_client=http_client
)

structured_llm = llm.with_structured_output(AnalisisTexto)

texto_prueba = "Me encantó la nueva película de acción, tiene muchos efectos especiales y emoción."

resultado = structured_llm.invoke(f"Analiza el siguiente texto: {texto_prueba}")

#print(type(resultado))
#print(dir(resultado))

print(resultado.model_dump_json())