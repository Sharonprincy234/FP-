import time
from typing import Dict, Any

class ResponseCache:
    """Simple response cache with timeout."""
    
    def __init__(self, timeout=300):
        self.cache: Dict[str, Dict] = {}
        self.timeout = timeout
        
    def get(self, key: str) -> Any:
        """Get a value from cache if it exists and is not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.timeout:
                return entry['value']
            else:
                # Remove expired entry
                del self.cache[key]
        return None
        
    def set(self, key: str, value: Any) -> None:
        """Set a value in cache."""
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
