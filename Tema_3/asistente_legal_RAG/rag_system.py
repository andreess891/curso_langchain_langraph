from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
import streamlit as st

from config import *
from prompts import *

def initialize_rag_system():
    # Vector Store
    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DB_PATH
    )

    # Modelos
    llm_queries = ChatOpenAI(model=QUERY_MODEL, temperature=0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0)

    # Retriever MMR (Maximal Marginal Relevance) para diversidad en los resultados de la búsqueda de documentos relevantes
    base_retriever = vector_store.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "fetch_k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )

    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever con prompt personalizado
    mmr_multi_retriever = MultiQueryRetriever.from_retriever(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # Funcion para formaterar y preprocesar los documentos recuperados antes de pasarlos al prompt
    def format_docs(docs):
        formatted = []
        
        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split('/')[-1] if '/' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Página: {doc.metadata['page']}"

            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")
        
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": mmr_multi_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, mmr_multi_retriever
    