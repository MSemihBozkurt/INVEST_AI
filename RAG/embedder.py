# embedder.py
from chromadb.utils import embedding_functions

def get_embedding_function(model_name="distiluse-base-multilingual-cased-v1"):
    """Initialize and return the embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)