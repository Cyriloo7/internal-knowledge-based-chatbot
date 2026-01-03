"""
Vector store optimization utilities
"""
import hashlib
import json
from functools import lru_cache
from typing import List, Dict, Any
from utils.cache import cache, make_cache_key

# Cache for vector store queries
VECTOR_CACHE_TIMEOUT = 3600  # 1 hour for vector searches

@lru_cache(maxsize=100)
def get_optimized_retriever(collection_name: str, k: int = 5):
    """Get cached retriever instance"""
    from client.src.components.retriever import Retriever
    return Retriever(collection_name=collection_name)

def cache_vector_search(query: str, collection_name: str, results: List[Dict[str, Any]], k: int = 5):
    """Cache vector search results"""
    cache_key = f"vector_search_{make_cache_key(query, collection=collection_name)}_k{k}"
    cache.set(cache_key, results, timeout=VECTOR_CACHE_TIMEOUT)
    return results

def get_cached_vector_search(query: str, collection_name: str, k: int = 5):
    """Get cached vector search results"""
    cache_key = f"vector_search_{make_cache_key(query, collection=collection_name)}_k{k}"
    return cache.get(cache_key)

def optimized_vector_search(query: str, collection_name: str, k: int = 5, use_cache: bool = True):
    """Optimized vector search with caching"""
    # Check cache first
    if use_cache:
        cached_results = get_cached_vector_search(query, collection_name, k)
        if cached_results is not None:
            return cached_results
    
    # Perform search
    retriever = get_optimized_retriever(collection_name, k)
    vectorstore = retriever.get_vector_store()
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    # Format results
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            'content': doc.page_content,
            'metadata': doc.metadata,
            'score': float(score)
        })
    
    # Cache results
    if use_cache:
        cache_vector_search(query, collection_name, formatted_results, k)
    
    return formatted_results

def invalidate_vector_cache(collection_name: str = None):
    """Invalidate vector search cache for a collection"""
    # For Redis, we would use pattern matching
    # For simple cache, we can't do pattern invalidation easily
    # This would need to be implemented with a cache key registry
    pass

def optimize_chromadb_settings():
    """Optimize ChromaDB settings for better performance"""
    import chromadb
    from chromadb.config import Settings
    
    # Optimized ChromaDB settings
    optimized_settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        # Performance optimizations
        chroma_db_impl="duckdb+parquet",  # Faster than sqlite
        persist_directory="./chroma_vectorstore"
    )
    
    return optimized_settings

