# Configuracion de los modelos
EMBEDDING_MODEL = "text-embedding-3-large"
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o"

# CONFIGURACION DEL VECTOR STORE
CHROMA_DB_PATH = "chroma_database"

# Configuracion del tipo de búsqueda en el retriever
SEARCH_TYPE = "mmr"  # Maximal Marginal Relevance para diversidad en los resultados
MMR_DIVERSITY_LAMBDA = 0.7  # Parámetro de diversidad para MMR (0.0 a 1.0, donde 1.0 es máxima diversidad)
MMR_FETCH_K = 20  # Número de documentos a recuperar antes de aplicar MMR
SEARCH_K = 2  # Número de documentos a recuperar para búsqueda tradicional (si no se usa MMR)
