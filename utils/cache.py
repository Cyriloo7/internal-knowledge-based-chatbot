"""
Intelligent caching utilities for RAG Chatbot
"""
from functools import wraps
import hashlib
import json
from datetime import datetime, timedelta
from flask_caching import Cache
import os

# Initialize cache
cache_config = {
    'CACHE_TYPE': 'simple',  # Use 'redis' for production with Redis
    'CACHE_DEFAULT_TIMEOUT': 300,  # 5 minutes default
}

# For production, use Redis if available
if os.getenv('REDIS_URL'):
    cache_config.update({
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_URL': os.getenv('REDIS_URL'),
        'CACHE_KEY_PREFIX': 'rag_chatbot_',
    })

cache = Cache(config=cache_config)

def make_cache_key(query, user_id=None, collection=None):
    """Generate a unique cache key from query parameters"""
    key_data = {
        'query': query.lower().strip(),
        'user_id': user_id,
        'collection': collection
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()

def cache_response(timeout=300, key_prefix='response'):
    """Decorator to cache function responses"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}_{make_cache_key(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = f(*args, **kwargs)
            cache.set(cache_key, result, timeout=timeout)
            return result
        return decorated_function
    return decorator

def invalidate_cache_pattern(pattern):
    """Invalidate all cache keys matching a pattern"""
    # For simple cache, we can't do pattern matching
    # For Redis, we would use: cache.cache.delete_pattern(pattern)
    pass

def get_cache_stats():
    """Get cache statistics"""
    try:
        if hasattr(cache.cache, 'info'):
            return cache.cache.info()
        return {'type': cache_config['CACHE_TYPE']}
    except:
        return {'type': 'unknown'}

