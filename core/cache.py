import hashlib
import json
import logging
import threading
import time
from typing import Any, Optional
from cachetools import TTLCache
from config.settings import settings

logger = logging.getLogger("cache_adapter")


class CacheAdapter:
    """
    Thread-safe dual-mode caching layer.
    Automatically connects to Redis if `REDIS_URL` is set; otherwise defaults to an
    in-memory TTL cache protected by a thread lock (threading.Lock).
    """
    def __init__(self):
        self.redis_client = None
        self.ttl_seconds = settings.CACHE_TTL_SECONDS
        self._lock = threading.Lock()

        if settings.REDIS_URL:
            try:
                import redis
                self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                # someitmes it gives bytes , With decode_responses=True, you'll get: '{"name": "Akanshu"}'

                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis production cache: %s", settings.REDIS_URL)
            except Exception as exc:
                logger.warning("Failed to connect to Redis (%s). Falling back to In-Memory cache.", exc)
                self.redis_client = None

        if self.redis_client is None:
            # In-Memory Cache: holds up to 1000 items with expiration
            self.memory_cache = TTLCache(maxsize=1000, ttl=self.ttl_seconds)
            logger.info("Initialized local In-Memory LRU cache (TTL: %d seconds).", self.ttl_seconds)

    @staticmethod
    def generate_key(user_id: str, query: str) -> str:
        """
        Normalizes and hashes user query into a consistent cache fingerprint.
        """
        normalized = " ".join(query.lower().strip().split()) # This increases cache hits rate.
        raw_key = f"{user_id}:{normalized}"
        fingerprint = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"plan_cache:{fingerprint}"

    def get(self, key: str) -> Optional[Any]:
        """Retrieves cached payload if valid and unexpired."""
        if self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception as exc:
                logger.error("Redis get failed: %s", exc)
                return None
        else:
            with self._lock:
                return self.memory_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stores payload in cache with TTL."""
        expiry = ttl if ttl is not None else self.ttl_seconds
        if self.redis_client:
            try:
                self.redis_client.setex(key, expiry, json.dumps(value))
            except Exception as exc:
                logger.error("Redis set failed: %s", exc)
        else:
            with self._lock:
                self.memory_cache[key] = value

    def clear(self, pattern: str = "plan_cache:*") -> None:
        """
        Clears cache entries.
        In Redis, safely deletes only keys matching the pattern (default: 'plan_cache:*')
        using scan_iter to prevent blocking and avoid wiping unrelated shared database data.
        """
        if self.redis_client:
            try:
                keys_to_delete = list(self.redis_client.scan_iter(match=pattern))
                if keys_to_delete:
                    self.redis_client.delete(*keys_to_delete)
            except Exception as exc:
                logger.error("Redis clear failed: %s", exc)
        else:
            with self._lock:
                self.memory_cache.clear()


# Global singleton instance
cache = CacheAdapter()
