# Analisis de sentimientos con un RunnableLambda
# ARQUITECTURA
# Texto de entrada → Preprocesamiento → Análisis Completo → Resultado
#                                           ↙        ↘
#                                    Resumen    Sentimiento

import json
import httpx

from dotenv import load_dotenv,find_dotenv

from langchain_core.runnables import RunnableLambda,RunnableParallel
from langchain_openai import ChatOpenAI


load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

# Configuración del modelo
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0, # Temperatura en cero debido a que para analisis de sentimientos no queremos respuestas creativas, sino respuestas consistentes y basadas en el texto de entrada.
    http_client=http_client
)

# Función de preprocesamiento
def preprocess_text(text):
    """Limpia el texto eliminando espacios extras y limitando longitud"""
    return text.strip()[:500]

# Convertir la función de preprocesamiento en un Runnable
preprocessor = RunnableLambda(preprocess_text)

# función para generar resumen
def generate_summary(text):
    """Genera un resumen conciso del texto"""
    prompt = f"Resume en una sola oración: {text}"
    response = llm.invoke(prompt)
    return response.content

# Convertir la función de generación de resumen en un Runnable
summary_brach = RunnableLambda(generate_summary)

# Análisis de sentimiento con formato JSON
def analyze_sentiment(text):
    """Analiza el sentimiento y devuelve resultado estructurado"""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde ÚNICAMENTE en formato JSON válido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}
    
    Texto: {text}"""
    
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en análisis"}

# Convertir la función de análisis de sentimiento en un Runnable
sentiment_branch = RunnableLambda(analyze_sentiment)

# Función para combinar resultados
def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado"""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"]
    }

# Convertir la función de combinación de resultados en un Runnable
merger = RunnableLambda(merge_results)

# Ejecutar ambas ramas en paralelo
parallel_analysis = RunnableParallel({
    "resumen": summary_brach,
    "sentimiento_data": sentiment_branch
})

# Crear la cadena completa de procesamiento
chain = preprocessor | parallel_analysis | merger
