import httpx

from dotenv import load_dotenv,find_dotenv
import numpy as np

from langchain_openai import OpenAIEmbeddings

load_dotenv(find_dotenv())
http_client = httpx.Client(verify=False)

# Crear una instancia de OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", http_client=http_client
)

texto1 = "La capital de Francia es París."
texto2 = "París es la ciuidad capital de Francia."

# Obtener los vectores de embeddings para ambos textos
vector1 = embeddings.embed_query(texto1)
vector2 = embeddings.embed_query(texto2)

print(f"Dimensiones del vector 1: {len(vector1)}")
print(f"Dimensiones del vector 2: {len(vector2)}")

# Calcular la similitud coseno entre los dos vectores a traves de la fórmula: cos_sim = (A . B) / (||A|| * ||B||)
# Donde A y B son los vectores de embeddings, "." representa el producto punto, y ||A|| y ||B|| son las normas (magnitudes) de los vectores.
# Tambien se puede usar la función de similitud coseno de sklearn.metrics.pairwise.cosine_similarity, pero aquí lo calculamos manualmente para entender el proceso.
cos_sim = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

print(f"Similitud coseno entre los dos textos: {cos_sim:.3f}")