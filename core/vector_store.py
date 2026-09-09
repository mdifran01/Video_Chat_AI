from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os


EMBEDDING_MODEL = "nomic-embed-text"

VECTOR_DB_DIR = "data/vectorstore"


def create_vector_store(rag_chunks:list):
    print("Creating Embiddings ....")
    
    embiddings=OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )
    vector_store=Chroma.from_documents(
        documents=rag_chunks,
        embedding=embiddings,
        persist_directory=VECTOR_DB_DIR
    )
    print("Documents stored in ChromaDB.")

    return vector_store