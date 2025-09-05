import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from functools import wraps
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncCache:
    """Simple in-memory cache for async operations"""
    def __init__(self, ttl: int = 300):  # 5 minutes default TTL
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }

# Global cache instance
cache = AsyncCache()

def async_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for async retry logic"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator

async def run_with_timeout(coro, timeout: float):
    """Run coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout} seconds")
        raise

def json_serializable(obj):
    """Make object JSON serializable"""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)

class TaskProgress:
    """Track progress of async tasks"""
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
    
    def add_task(self, task_id: str, description: str, weight: float = 1.0):
        self.tasks[task_id] = {
            'description': description,
            'weight': weight,
            'status': 'pending',
            'start_time': None,
            'end_time': None,
            'result': None,
            'error': None
        }
    
    def start_task(self, task_id: str):
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = 'running'
            self.tasks[task_id]['start_time'] = time.time()
    
    def complete_task(self, task_id: str, result: Any = None):
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = 'completed'
            self.tasks[task_id]['end_time'] = time.time()
            self.tasks[task_id]['result'] = result
    
    def fail_task(self, task_id: str, error: Exception):
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = 'failed'
            self.tasks[task_id]['end_time'] = time.time()
            self.tasks[task_id]['error'] = str(error)
    
    def get_progress(self) -> Dict[str, Any]:
        total_weight = sum(task['weight'] for task in self.tasks.values())
        completed_weight = sum(
            task['weight'] for task in self.tasks.values() 
            if task['status'] == 'completed'
        )
        
        return {
            'total_tasks': len(self.tasks),
            'completed_tasks': len([t for t in self.tasks.values() if t['status'] == 'completed']),
            'failed_tasks': len([t for t in self.tasks.values() if t['status'] == 'failed']),
            'running_tasks': len([t for t in self.tasks.values() if t['status'] == 'running']),
            'progress_percentage': (completed_weight / total_weight * 100) if total_weight > 0 else 0,
            'tasks': self.tasks
        }

# Global progress tracker
progress_tracker = TaskProgress()