"""
Production-grade caching layer for TriagePlus
Supports in-memory and Redis-backed caching with TTL
"""

import json
import time
import hashlib
from typing import Any, Optional, Callable, TypeVar, Dict
from functools import wraps
import asyncio
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheManager:
    """Multi-tier caching system"""
    
    def __init__(self, use_redis: bool = False):
        self.use_redis = use_redis
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_client = None
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis unavailable, falling back to memory cache: {e}")
                self.use_redis = False
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function args and kwargs"""
        key_parts = [prefix]
        
        # Add stringified args
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
        
        # Add sorted kwargs
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set cache value with TTL"""
        try:
            data = {
                'value': value,
                'expires_at': datetime.utcnow().timestamp() + ttl_seconds
            }
            
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.setex(
                        key,
                        ttl_seconds,
                        json.dumps(data)
                    )
                    return True
                except Exception as e:
                    logger.error(f"Redis set failed: {e}, falling back to memory")
            
            # Memory cache
            self.memory_cache[key] = data
            return True
        
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            if self.use_redis and self.redis_client:
                try:
                    value_json = self.redis_client.get(key)
                    if value_json:
                        data = json.loads(value_json)
                        if data['expires_at'] > datetime.utcnow().timestamp():
                            return data['value']
                        else:
                            self.redis_client.delete(key)
                except Exception as e:
                    logger.error(f"Redis get failed: {e}, trying memory")
            
            # Memory cache
            if key in self.memory_cache:
                data = self.memory_cache[key]
                if data['expires_at'] > datetime.utcnow().timestamp():
                    return data['value']
                else:
                    del self.memory_cache[key]
            
            return None
        
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        try:
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.delete(key)
                except Exception as e:
                    logger.error(f"Redis delete failed: {e}")
            
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            return True
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache"""
        try:
            self.memory_cache.clear()
            
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.error(f"Redis flush failed: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False

# Global cache instance
_cache_manager = None

def get_cache_manager(use_redis: bool = False) -> CacheManager:
    """Get or create cache manager singleton"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(use_redis=use_redis)
    return _cache_manager

def cached(ttl_seconds: int = 3600, prefix: str = "cache"):
    """Decorator for caching function results"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            cache = get_cache_manager()
            cache_key = cache._generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cached_value
            
            # Cache miss, call function
            logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl_seconds)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            cache = get_cache_manager()
            cache_key = cache._generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cached_value
            
            # Cache miss, call function
            logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl_seconds)
            return result
        
        # Return appropriate wrapper
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
