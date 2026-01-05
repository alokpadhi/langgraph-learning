# Chapter 18: Production Engineering for LangGraph Agents

## 🏭 Introduction: Building Production-Ready Systems

Moving from prototype to production requires more than just functional code. **Production engineering** ensures systems are:
- **Reliable**: Handle errors gracefully, recover automatically
- **Performant**: Fast response times, efficient resource use
- **Observable**: Monitor health, diagnose issues quickly
- **Scalable**: Handle increasing load
- **Maintainable**: Easy to update and debug

Think of it like the difference between a **prototype car** and a **production vehicle** - the prototype proves the concept, but the production version needs safety features, reliability testing, maintenance plans, and scale manufacturing.

---

## 🛡️ Part 1: Error Handling and Recovery Strategies

### Theory: Production Error Handling

#### Types of Errors in Agent Systems

**1. Transient Errors (Recoverable):**
```
- Network timeouts
- API rate limits
- Temporary service unavailability
- Database connection issues

Strategy: Retry with backoff
```

**2. Permanent Errors (Non-recoverable):**
```
- Invalid API keys
- Malformed input data
- Authorization failures
- Business logic violations

Strategy: Fail fast, log, alert
```

**3. Partial Failures:**
```
- Some nodes succeed, others fail
- Multi-agent system with failed agents
- Degraded functionality

Strategy: Graceful degradation
```

**4. Unexpected Errors:**
```
- Unhandled exceptions
- Out of memory
- Infinite loops
- Assertion failures

Strategy: Circuit breakers, timeouts
```

#### Error Handling Strategies

**Strategy 1: Retry with Exponential Backoff**

```python
import time
import random

def retry_with_backoff(
    func,
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0
):
    """Retry with exponential backoff"""
    
    for attempt in range(max_retries):
        try:
            return func()
        except TransientError as e:
            if attempt == max_retries - 1:
                raise  # Last attempt failed
            
            # Calculate delay: base * 2^attempt + jitter
            delay = min(
                base_delay * (2 ** attempt) + random.uniform(0, 1),
                max_delay
            )
            
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s")
            time.sleep(delay)
```

**Why exponential backoff?**
- Linear backoff: Might retry too quickly
- Exponential: Gives service time to recover
- Jitter: Prevents thundering herd problem

**Strategy 2: Circuit Breaker**

```python
class CircuitBreaker:
    """Circuit breaker pattern"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen("Circuit breaker is open")
        
        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        self.failures = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
```

**Why circuit breakers?**
- Prevent cascading failures
- Give failing service time to recover
- Fast-fail instead of waiting for timeout

**Strategy 3: Fallback Mechanisms**

```python
def with_fallback(primary_func, fallback_func):
    """Try primary, use fallback if fails"""
    try:
        return primary_func()
    except Exception as e:
        logger.warning(f"Primary failed: {e}, using fallback")
        return fallback_func()

# Example
result = with_fallback(
    lambda: expensive_ai_model(prompt),
    lambda: simple_rule_based_fallback(prompt)
)
```

**Strategy 4: Timeout Protection**

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

def with_timeout(func, timeout_seconds):
    """Execute function with timeout"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func()
        signal.alarm(0)  # Cancel alarm
        return result
    except TimeoutError:
        logger.error(f"Function timed out after {timeout_seconds}s")
        raise
```

**Strategy 5: Graceful Degradation**

```python
def get_recommendations(user_id):
    """Get recommendations with graceful degradation"""
    
    try:
        # Try ML-based recommendations
        return ml_recommender.predict(user_id)
    except MLModelError:
        logger.warning("ML model failed, falling back to collaborative filtering")
        try:
            return collaborative_filter(user_id)
        except CollaborativeFilterError:
            logger.warning("Collaborative filter failed, using popular items")
            return get_popular_items()  # Always works
```

#### Error Context and Logging

**What to log when errors occur:**

```python
error_context = {
    "timestamp": datetime.utcnow().isoformat(),
    "error_type": type(e).__name__,
    "error_message": str(e),
    "stack_trace": traceback.format_exc(),
    
    # Context
    "workflow_id": state.get("workflow_id"),
    "node_name": current_node,
    "state_snapshot": sanitize_state(state),  # Remove sensitive data
    
    # Environment
    "python_version": sys.version,
    "dependencies": get_package_versions(),
    
    # Request info
    "user_id": user_id,
    "session_id": session_id,
    "request_id": request_id
}

logger.error(f"Error in {current_node}", extra=error_context)
```

#### Recovery Strategies

**1. Automatic Retry:**
```
Error occurs → Retry immediately or with backoff → Success or give up
```

**2. Checkpoint and Resume:**
```
Error occurs → Save state → Later: Resume from checkpoint
```

**3. Human Intervention:**
```
Error occurs → Notify human → Human fixes → Resume
```

**4. Alternate Path:**
```
Error occurs → Route to fallback node → Continue with degraded function
```

**5. Compensating Transactions:**
```
Error occurs → Undo previous steps → Return to clean state
```

---

### Implementation: Robust Error Handling

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional, Callable
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import time
import random
import logging
from datetime import datetime
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ERROR HANDLING UTILITIES ====================

class RetryableError(Exception):
    """Error that should be retried"""
    pass

class PermanentError(Exception):
    """Error that should not be retried"""
    pass

class CircuitBreakerOpen(Exception):
    """Circuit breaker is open"""
    pass

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
):
    """Decorator for retry with exponential backoff"""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                
                except PermanentError:
                    # Don't retry permanent errors
                    raise
                
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        logger.error(f"All {max_retries} attempts failed")
                        raise
                    
                    # Calculate delay with jitter
                    delay = min(
                        base_delay * (backoff_factor ** attempt) + random.uniform(0, 1),
                        max_delay
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker is OPEN. "
                    f"Will retry after {self.recovery_timeout}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker CLOSED (recovered)")
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
            logger.warning("Circuit breaker OPEN (half-open attempt failed)")
        
        elif self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                f"Circuit breaker OPEN after {self.failure_count} failures"
            )

def with_fallback(primary: Callable, fallback: Callable):
    """Execute primary function with fallback on error"""
    
    def wrapper(*args, **kwargs):
        try:
            return primary(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"Primary function failed: {e}. Using fallback."
            )
            try:
                return fallback(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise
    
    return wrapper

# ==================== PRODUCTION STATE ====================

class ProductionState(TypedDict):
    """State with error handling metadata"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    result: str
    error_count: Annotated[int, add]
    retry_count: Annotated[int, add]
    errors: Annotated[List[Dict], add]
    circuit_breaker_status: str

# ==================== SIMULATED UNRELIABLE SERVICES ====================

class UnreliableService:
    """Simulates an unreliable external service"""
    
    def __init__(self, failure_rate: float = 0.5):
        self.failure_rate = failure_rate
        self.call_count = 0
    
    def call(self):
        """Call service (may fail randomly)"""
        self.call_count += 1
        
        if random.random() < self.failure_rate:
            raise RetryableError(f"Service call {self.call_count} failed (simulated)")
        
        return f"Success on call {self.call_count}"

unreliable_service = UnreliableService(failure_rate=0.6)
circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

# ==================== PRODUCTION NODES ====================

@retry_with_backoff(max_retries=3, base_delay=0.5)
def resilient_processing(state: ProductionState) -> dict:
    """Node with retry logic"""
    logger.info("🔄 Resilient processing with retry...")
    
    try:
        # Simulate unreliable operation
        result = unreliable_service.call()
        
        logger.info(f"✅ Success: {result}")
        
        return {
            "result": result,
            "messages": [AIMessage(content=f"Processing succeeded: {result}")]
        }
    
    except Exception as e:
        logger.error(f"❌ Processing failed after retries: {e}")
        
        error_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "node": "resilient_processing"
        }
        
        return {
            "error_count": 1,
            "errors": [error_info],
            "messages": [AIMessage(content=f"Processing failed: {e}")]
        }

def circuit_breaker_protected(state: ProductionState) -> dict:
    """Node with circuit breaker protection"""
    logger.info("⚡ Circuit breaker protected operation...")
    
    try:
        result = circuit_breaker.call(unreliable_service.call)
        
        return {
            "result": result,
            "circuit_breaker_status": circuit_breaker.state,
            "messages": [AIMessage(content=f"CB protected call succeeded: {result}")]
        }
    
    except CircuitBreakerOpen as e:
        logger.warning(f"⚠️ Circuit breaker is open: {e}")
        
        return {
            "circuit_breaker_status": "OPEN",
            "messages": [AIMessage(content="Circuit breaker is open - using fallback")]
        }
    
    except Exception as e:
        logger.error(f"❌ Call failed: {e}")
        
        return {
            "error_count": 1,
            "circuit_breaker_status": circuit_breaker.state,
            "errors": [{
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }],
            "messages": [AIMessage(content=f"Call failed: {e}")]
        }

def fallback_handler(state: ProductionState) -> dict:
    """Fallback node for graceful degradation"""
    logger.info("🔄 Executing fallback handler...")
    
    # Provide degraded but functional result
    fallback_result = "Fallback result (degraded functionality)"
    
    return {
        "result": fallback_result,
        "messages": [AIMessage(content=f"Using fallback: {fallback_result}")]
    }

def error_recovery(state: ProductionState) -> dict:
    """Recover from errors"""
    logger.info("🛠️ Attempting error recovery...")
    
    if state["error_count"] > 0:
        logger.warning(f"Encountered {state['error_count']} errors")
        
        # Check if errors are recoverable
        if state["error_count"] < 3:
            return {
                "messages": [AIMessage(content="Errors detected but continuing...")]
            }
        else:
            return {
                "messages": [AIMessage(content="Too many errors - escalating...")]
            }
    
    return {
        "messages": [AIMessage(content="No errors to recover from")]
    }

def generate_report(state: ProductionState) -> dict:
    """Generate final report"""
    
    report = f"""
PRODUCTION WORKFLOW REPORT
{'='*60}

Task: {state['task']}
Result: {state.get('result', 'No result')}

Error Statistics:
- Total Errors: {state['error_count']}
- Retry Attempts: {state['retry_count']}
- Circuit Breaker Status: {state.get('circuit_breaker_status', 'N/A')}

Error Details:
{chr(10).join([f"  - {e['timestamp']}: {e.get('error_message', e.get('error', 'Unknown'))}" for e in state.get('errors', [])])}

Status: {'Success with degradation' if state['error_count'] > 0 else 'Success'}
"""
    
    return {
        "messages": [AIMessage(content=report)]
    }

# ==================== BUILD PRODUCTION WORKFLOW ====================

production_workflow = StateGraph(ProductionState)

production_workflow.add_node("process", resilient_processing)
production_workflow.add_node("circuit_breaker", circuit_breaker_protected)
production_workflow.add_node("fallback", fallback_handler)
production_workflow.add_node("recovery", error_recovery)
production_workflow.add_node("report", generate_report)

production_workflow.set_entry_point("process")
production_workflow.add_edge("process", "circuit_breaker")
production_workflow.add_edge("circuit_breaker", "recovery")
production_workflow.add_edge("recovery", "report")
production_workflow.add_edge("report", END)

production_system = production_workflow.compile()

logger.info("✅ Production workflow compiled")

# ==================== DEMONSTRATION ====================

def demonstrate_error_handling():
    """Demonstrate production error handling"""
    
    print("\n" + "="*60)
    print("PRODUCTION ERROR HANDLING DEMONSTRATION")
    print("="*60)
    print("\nThis demo uses simulated unreliable services")
    print("that fail 60% of the time to show error handling.\n")
    
    for run in range(3):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}")
        print(f"{'='*60}")
        
        # Reset services
        unreliable_service.call_count = 0
        
        initial_state = {
            "messages": [HumanMessage(content="Start production workflow")],
            "task": f"Production task {run + 1}",
            "result": "",
            "error_count": 0,
            "retry_count": 0,
            "errors": [],
            "circuit_breaker_status": "CLOSED"
        }
        
        try:
            result = production_system.invoke(initial_state)
            
            print(f"\n{result['messages'][-1].content}")
            
        except Exception as e:
            print(f"\n❌ Workflow failed completely: {e}")
        
        time.sleep(1)  # Brief pause between runs

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    demonstrate_error_handling()
```

---

## ⚡ Part 2: Performance Optimization and Caching

### Theory: Performance Optimization

#### Performance Bottlenecks in Agent Systems

**1. LLM API Calls:**
```
Problem: Slow (1-5 seconds per call), expensive
Impact: Major bottleneck in multi-node workflows
Solution: Caching, batching, parallel calls
```

**2. Data Processing:**
```
Problem: Large datasets, complex transformations
Impact: Memory usage, processing time
Solution: Streaming, chunking, efficient algorithms
```

**3. External API Calls:**
```
Problem: Network latency, rate limits
Impact: Workflow delays
Solution: Caching, connection pooling, async
```

**4. State Management:**
```
Problem: Large state objects, frequent updates
Impact: Memory, serialization overhead
Solution: State compression, incremental updates
```

#### Caching Strategies

**Strategy 1: Response Caching**

```python
from functools import lru_cache
import hashlib
import json

class ResponseCache:
    """Cache for LLM responses"""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def _hash_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key"""
        key_data = {"prompt": prompt, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """Get cached response"""
        key = self._hash_key(prompt, **kwargs)
        return self.cache.get(key)
    
    def set(self, prompt: str, response: str, **kwargs):
        """Cache response"""
        if len(self.cache) >= self.max_size:
            # Evict oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))
        
        key = self._hash_key(prompt, **kwargs)
        self.cache[key] = response

# Usage
cache = ResponseCache()

def call_llm_with_cache(prompt: str) -> str:
    cached = cache.get(prompt)
    if cached:
        logger.info("Cache hit!")
        return cached
    
    response = llm.invoke(prompt)
    cache.set(prompt, response)
    return response
```

**Why caching works:**
- Many queries are repeated
- LLM responses are deterministic (with temperature=0)
- Saves time and money

**Strategy 2: Semantic Caching**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCache:
    """Cache based on semantic similarity"""
    
    def __init__(self, similarity_threshold=0.95):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = []
        self.responses = []
        self.threshold = similarity_threshold
    
    def get(self, prompt: str) -> Optional[str]:
        """Get cached response for semantically similar prompt"""
        if not self.embeddings:
            return None
        
        prompt_embedding = self.model.encode(prompt)
        
        # Find most similar cached prompt
        similarities = [
            np.dot(prompt_embedding, cached_emb) /
            (np.linalg.norm(prompt_embedding) * np.linalg.norm(cached_emb))
            for cached_emb in self.embeddings
        ]
        
        max_similarity = max(similarities)
        
        if max_similarity >= self.threshold:
            idx = similarities.index(max_similarity)
            logger.info(f"Semantic cache hit! Similarity: {max_similarity:.3f}")
            return self.responses[idx]
        
        return None
    
    def set(self, prompt: str, response: str):
        """Cache response with embedding"""
        embedding = self.model.encode(prompt)
        self.embeddings.append(embedding)
        self.responses.append(response)
```

**Why semantic caching:**
- Different phrasings of same question
- Variations that mean the same thing
- More cache hits than exact matching

**Strategy 3: Tiered Caching**

```python
class TieredCache:
    """Multi-level cache: memory → disk → remote"""
    
    def __init__(self):
        self.memory_cache = {}  # Fast, limited size
        self.disk_cache_path = "/tmp/cache"  # Slower, larger
        # self.redis_cache = Redis()  # Shared across instances
    
    def get(self, key: str) -> Optional[str]:
        # Level 1: Memory
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Level 2: Disk
        disk_value = self._get_from_disk(key)
        if disk_value:
            self.memory_cache[key] = disk_value  # Promote to memory
            return disk_value
        
        # Level 3: Remote (Redis)
        # remote_value = self.redis_cache.get(key)
        # if remote_value:
        #     self.memory_cache[key] = remote_value
        #     return remote_value
        
        return None
    
    def set(self, key: str, value: str):
        self.memory_cache[key] = value
        self._set_to_disk(key, value)
        # self.redis_cache.set(key, value)
```

#### Batching and Parallelization

**Batching:**
```python
def batch_process(items: List[str], batch_size: int = 10):
    """Process items in batches"""
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch together (more efficient)
        results = llm.batch(batch)
        
        for result in results:
            yield result
```

**Parallel Processing:**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_process(items: List[str], max_workers: int = 5):
    """Process items in parallel"""
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_item, item): item
            for item in items
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                yield result
            except Exception as e:
                logger.error(f"Parallel processing error: {e}")
```

#### Database Query Optimization

**1. Connection Pooling:**
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@host/db',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

**2. Query Optimization:**
```python
# Bad: N+1 query problem
for user in users:
    orders = db.query(Order).filter(Order.user_id == user.id).all()

# Good: Join and fetch all at once
users_with_orders = db.query(User).join(Order).all()
```

**3. Indexing:**
```sql
-- Add indexes on frequently queried columns
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_order_created ON orders(created_at);
```

#### Memory Optimization

**1. Streaming Instead of Loading All:**
```python
# Bad: Load everything into memory
all_data = load_entire_dataset()
for item in all_data:
    process(item)

# Good: Stream data
for item in stream_dataset():
    process(item)
```

**2. Generators Instead of Lists:**
```python
# Bad: Creates full list in memory
def get_numbers():
    return [i for i in range(1000000)]

# Good: Yields one at a time
def get_numbers():
    for i in range(1000000):
        yield i
```

---

### Implementation: Performance Optimization

```python
from typing import TypedDict, Annotated, Sequence, List, Optional
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import time
import hashlib
import json
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CACHING SYSTEM ====================

class SimpleCache:
    """Simple in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, *args, **kwargs) -> Optional[any]:
        """Get from cache"""
        key = self._make_key(*args, **kwargs)
        
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            logger.info(f"✅ Cache HIT (hit rate: {self.hit_rate:.1%})")
            return self.cache[key]
        
        self.misses += 1
        logger.info(f"❌ Cache MISS (hit rate: {self.hit_rate:.1%})")
        return None
    
    def set(self, value: any, *args, **kwargs):
        """Set in cache"""
        key = self._make_key(*args, **kwargs)
        
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            logger.info(f"🗑️ Evicted LRU entry")
        
        self.cache[key] = value
        self.access_order.append(key)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def stats(self) -> dict:
        """Get cache statistics"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size
        }

# Global cache
response_cache = SimpleCache(max_size=50)

def cached(cache_instance: SimpleCache):
    """Decorator to cache function results"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache
            cached_result = cache_instance.get(*args, **kwargs)
            if cached_result is not None:
                return cached_result
            
            # Cache miss - compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_instance.set(result, *args, **kwargs)
            
            return result
        
        return wrapper
    return decorator

# ==================== PERFORMANCE MONITORING ====================

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics = []
    
    def record(self, operation: str, duration: float, **metadata):
        """Record performance metric"""
        self.metrics.append({
            "timestamp": time.time(),
            "operation": operation,
            "duration_ms": duration * 1000,
            **metadata
        })
    
    def get_stats(self, operation: Optional[str] = None) -> dict:
        """Get performance statistics"""
        
        if operation:
            relevant = [m for m in self.metrics if m["operation"] == operation]
        else:
            relevant = self.metrics
        
        if not relevant:
            return {}
        
        durations = [m["duration_ms"] for m in relevant]
        
        return {
            "count": len(durations),
            "mean_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "total_ms": sum(durations)
        }
    
    def report(self) -> str:
        """Generate performance report"""
        
        operations = set(m["operation"] for m in self.metrics)
        
        lines = ["PERFORMANCE REPORT", "=" * 60, ""]
        
        for operation in operations:
            stats = self.get_stats(operation)
            lines.append(f"{operation}:")
            lines.append(f"  Count: {stats['count']}")
            lines.append(f"  Mean: {stats['mean_ms']:.1f}ms")
            lines.append(f"  Min: {stats['min_ms']:.1f}ms")
            lines.append(f"  Max: {stats['max_ms']:.1f}ms")
            lines.append(f"  Total: {stats['total_ms']:.1f}ms")
            lines.append("")
        
        # Cache stats
        cache_stats = response_cache.stats()
        lines.append("Cache Statistics:")
        lines.append(f"  Hits: {cache_stats['hits']}")
        lines.append(f"  Misses: {cache_stats['misses']}")
        lines.append(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
        lines.append(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
        
        return "\n".join(lines)

perf_monitor = PerformanceMonitor()

def timed(operation_name: str):
    """Decorator to time function execution"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                perf_monitor.record(
                    operation_name,
                    duration,
                    success=True
                )
                
                logger.info(f"⏱️ {operation_name} took {duration*1000:.1f}ms")
                
                return result
            
            except Exception as e:
                duration = time.time() - start
                
                perf_monitor.record(
                    operation_name,
                    duration,
                    success=False,
                    error=str(e)
                )
                
                raise
        
        return wrapper
    return decorator

# ==================== OPTIMIZED STATE ====================

class OptimizedState(TypedDict):
    """State for performance testing"""
    messages: Annotated[Sequence[BaseMessage], add]
    query: str
    result: str
    cache_used: bool

# ==================== OPTIMIZED NODES ====================

@timed("expensive_operation")
@cached(response_cache)
def expensive_operation(query: str) -> str:
    """Simulate expensive operation (with caching)"""
    logger.info(f"🐌 Performing expensive operation for: {query}")
    
    # Simulate slow operation
    time.sleep(2.0)
    
    return f"Processed result for: {query}"

def cached_node(state: OptimizedState) -> dict:
    """Node that uses caching"""
    logger.info("💾 Cached node processing...")
    
    query = state["query"]
    
    # This will be cached after first call
    result = expensive_operation(query)
    
    return {
        "result": result,
        "cache_used": True,
        "messages": [AIMessage(content=result)]
    }

@timed("fast_operation")
def fast_operation(state: OptimizedState) -> dict:
    """Fast operation for comparison"""
    logger.info("⚡ Fast operation processing...")
    
    time.sleep(0.1)
    
    return {
        "messages": [AIMessage(content="Fast operation complete")]
    }

def performance_summary(state: OptimizedState) -> dict:
    """Generate performance summary"""
    
    report = perf_monitor.report()
    
    return {
        "messages": [AIMessage(content=report)]
    }

# ==================== BUILD OPTIMIZED WORKFLOW ====================

optimized_workflow = StateGraph(OptimizedState)

optimized_workflow.add_node("cached", cached_node)
optimized_workflow.add_node("fast", fast_operation)
optimized_workflow.add_node("summary", performance_summary)

optimized_workflow.set_entry_point("cached")
optimized_workflow.add_edge("cached", "fast")
optimized_workflow.add_edge("fast", "summary")
optimized_workflow.add_edge("summary", END)

optimized_system = optimized_workflow.compile()

# ==================== DEMONSTRATION ====================

def demonstrate_caching():
    """Demonstrate caching benefits"""
    
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    print("\nDemonstrating cache effectiveness...")
    
    # Reset metrics
    perf_monitor.metrics.clear()
    response_cache.cache.clear()
    response_cache.hits = 0
    response_cache.misses = 0
    
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What is machine learning?",  # Repeat - should hit cache
        "Explain neural networks",     # Repeat - should hit cache
        "What is machine learning?"    # Repeat again - should hit cache
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*60}")
        
        state = {
            "messages": [HumanMessage(content="Start")],
            "query": query,
            "result": "",
            "cache_used": False
        }
        
        result = optimized_system.invoke(state)
        
        if i == len(queries):
            # Show final report on last query
            print(f"\n{result['messages'][-1].content}")

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    demonstrate_caching()
```

---
# Chapter 18: Production Engineering for LangGraph Agents (Continued)

## 📊 Part 3: Monitoring and Observability

### Theory: Production Monitoring

#### What Is Observability?

**Observability** is the ability to understand what's happening inside your system by examining its outputs. It consists of three pillars:

**1. Metrics** (What happened?)
```
- Request count
- Response time
- Error rate
- Resource usage
```

**2. Logs** (Detailed events)
```
- Workflow execution logs
- Error messages
- Debug information
- Audit trails
```

**3. Traces** (Distributed execution flow)
```
- Request path through system
- Time spent in each component
- Dependencies and relationships
```

#### Why Monitor Agent Systems?

**1. Detect Issues Early:**
```
Before: Users report problems
With Monitoring: Alert triggers before users notice
```

**2. Understand Behavior:**
```
- Which nodes take longest?
- Where do errors occur?
- What's the success rate?
```

**3. Capacity Planning:**
```
- How much load can system handle?
- When to scale up?
- Resource utilization patterns
```

**4. Cost Management:**
```
- LLM API costs
- Compute costs
- Storage costs
```

#### Key Metrics for Agent Systems

**System Health Metrics:**
```python
system_metrics = {
    # Availability
    "uptime_percent": 99.9,
    "health_check_status": "healthy",
    
    # Performance
    "avg_response_time_ms": 1250,
    "p95_response_time_ms": 2000,
    "p99_response_time_ms": 3500,
    
    # Throughput
    "requests_per_minute": 120,
    "workflows_completed": 1543,
    
    # Errors
    "error_rate_percent": 0.5,
    "errors_last_hour": 3
}
```

**Business Metrics:**
```python
business_metrics = {
    # Usage
    "active_users": 450,
    "workflows_per_user": 3.4,
    
    # Cost
    "llm_api_calls": 5430,
    "estimated_cost_usd": 12.45,
    
    # Quality
    "user_satisfaction_score": 4.2,
    "approval_rate_percent": 87.3
}
```

**Agent-Specific Metrics:**
```python
agent_metrics = {
    # Workflow execution
    "avg_nodes_per_workflow": 7.2,
    "avg_workflow_duration_ms": 8500,
    
    # Node performance
    "node_execution_times": {
        "analyzer": 1200,
        "generator": 3400,
        "reviewer": 900
    },
    
    # Human-in-loop
    "approval_wait_time_avg_ms": 45000,
    "approval_rate_percent": 92,
    
    # Caching
    "cache_hit_rate_percent": 73.5,
    "cache_size_mb": 245
}
```

#### Monitoring Levels

**Level 1: Application Monitoring**
```python
# Track application-level metrics
logger.info("workflow_started", extra={
    "workflow_id": "wf-123",
    "user_id": "user-456",
    "workflow_type": "data_processing"
})
```

**Level 2: Infrastructure Monitoring**
```python
# Track system resources
metrics = {
    "cpu_usage_percent": 45,
    "memory_usage_mb": 2048,
    "disk_io_mb_per_sec": 12,
    "network_throughput_mbps": 100
}
```

**Level 3: Business Monitoring**
```python
# Track business outcomes
business_event = {
    "event": "workflow_completed",
    "user_segment": "enterprise",
    "workflow_value": 50.00,  # Revenue impact
    "user_satisfaction": 5
}
```

#### Alerting Strategies

**Alert Types:**

**1. Threshold Alerts:**
```python
if error_rate > 5.0:
    alert("ERROR_RATE_HIGH", severity="critical")

if avg_response_time > 5000:
    alert("SLOW_RESPONSE", severity="warning")
```

**2. Anomaly Detection:**
```python
# Alert on unusual patterns
if current_requests < historical_average * 0.5:
    alert("TRAFFIC_DROP", severity="warning")

if error_rate > historical_p95:
    alert("UNUSUAL_ERROR_RATE", severity="warning")
```

**3. Composite Alerts:**
```python
# Multiple conditions
if error_rate > 10 and response_time > 10000:
    alert("SYSTEM_DEGRADED", severity="critical")
```

**Alert Severity Levels:**
```
CRITICAL: System down, immediate action required
WARNING: Degraded performance, investigate soon
INFO: Notable event, awareness only
```

#### Distributed Tracing

**Why tracing matters:**
```
Workflow: A → B → C → D
Without tracing: "System is slow" (where?)
With tracing: "Node C takes 80% of time" (actionable)
```

**Trace Structure:**
```python
trace = {
    "trace_id": "trace-123",
    "workflow_id": "wf-456",
    "total_duration_ms": 5430,
    "spans": [
        {
            "span_id": "span-1",
            "parent_id": None,
            "operation": "workflow_start",
            "duration_ms": 5430
        },
        {
            "span_id": "span-2",
            "parent_id": "span-1",
            "operation": "node_analyzer",
            "duration_ms": 1200
        },
        {
            "span_id": "span-3",
            "parent_id": "span-1",
            "operation": "node_generator",
            "duration_ms": 3400
        }
    ]
}
```

#### Logging Best Practices

**Structured Logging:**
```python
# Bad: Unstructured
logger.info(f"User {user_id} started workflow {workflow_id}")

# Good: Structured
logger.info("workflow_started", extra={
    "user_id": user_id,
    "workflow_id": workflow_id,
    "timestamp": datetime.utcnow().isoformat()
})
```

**Log Levels:**
```python
DEBUG: Detailed diagnostic info (dev only)
INFO: General informational messages
WARNING: Warning messages (potential issues)
ERROR: Error messages (handled errors)
CRITICAL: Critical errors (system failures)
```

**What to Log:**
```python
# Start/end of operations
logger.info("operation_started", extra={"operation": "analysis"})
logger.info("operation_completed", extra={"operation": "analysis", "duration_ms": 1234})

# State transitions
logger.info("state_changed", extra={"from": "pending", "to": "processing"})

# Errors with context
logger.error("operation_failed", extra={
    "operation": "api_call",
    "error": str(e),
    "retry_count": 3
})

# Business events
logger.info("user_action", extra={"action": "approved", "user_id": user_id})
```

---

### Implementation: Monitoring System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== METRICS COLLECTOR ====================

@dataclass
class Metrics:
    """Collects and stores metrics"""
    
    # Counters
    workflow_started: int = 0
    workflow_completed: int = 0
    workflow_failed: int = 0
    node_executions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Timings (in seconds)
    workflow_durations: List[float] = field(default_factory=list)
    node_durations: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Errors
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_node: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Business metrics
    user_approvals: int = 0
    user_rejections: int = 0
    
    def record_workflow_start(self):
        """Record workflow start"""
        self.workflow_started += 1
    
    def record_workflow_complete(self, duration: float):
        """Record workflow completion"""
        self.workflow_completed += 1
        self.workflow_durations.append(duration)
    
    def record_workflow_failed(self):
        """Record workflow failure"""
        self.workflow_failed += 1
    
    def record_node_execution(self, node_name: str, duration: float):
        """Record node execution"""
        self.node_executions[node_name] += 1
        self.node_durations[node_name].append(duration)
    
    def record_error(self, error_type: str, node_name: str):
        """Record error"""
        self.errors_by_type[error_type] += 1
        self.errors_by_node[node_name] += 1
    
    def record_approval(self, approved: bool):
        """Record user approval/rejection"""
        if approved:
            self.user_approvals += 1
        else:
            self.user_rejections += 1
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        total = self.workflow_completed + self.workflow_failed
        return (self.workflow_completed / total * 100) if total > 0 else 0.0
    
    def get_approval_rate(self) -> float:
        """Calculate approval rate"""
        total = self.user_approvals + self.user_rejections
        return (self.user_approvals / total * 100) if total > 0 else 0.0
    
    def get_avg_duration(self) -> float:
        """Get average workflow duration"""
        if not self.workflow_durations:
            return 0.0
        return sum(self.workflow_durations) / len(self.workflow_durations)
    
    def get_node_stats(self, node_name: str) -> dict:
        """Get statistics for a specific node"""
        durations = self.node_durations.get(node_name, [])
        
        if not durations:
            return {}
        
        return {
            "count": self.node_executions[node_name],
            "avg_duration_ms": sum(durations) / len(durations) * 1000,
            "min_duration_ms": min(durations) * 1000,
            "max_duration_ms": max(durations) * 1000,
            "errors": self.errors_by_node.get(node_name, 0)
        }
    
    def generate_report(self) -> str:
        """Generate metrics report"""
        
        lines = [
            "MONITORING REPORT",
            "=" * 60,
            "",
            "Workflow Metrics:",
            f"  Started: {self.workflow_started}",
            f"  Completed: {self.workflow_completed}",
            f"  Failed: {self.workflow_failed}",
            f"  Success Rate: {self.get_success_rate():.1f}%",
            f"  Avg Duration: {self.get_avg_duration()*1000:.1f}ms",
            ""
        ]
        
        if self.node_executions:
            lines.append("Node Execution Metrics:")
            for node_name in sorted(self.node_executions.keys()):
                stats = self.get_node_stats(node_name)
                lines.append(f"  {node_name}:")
                lines.append(f"    Executions: {stats['count']}")
                lines.append(f"    Avg Duration: {stats['avg_duration_ms']:.1f}ms")
                lines.append(f"    Errors: {stats['errors']}")
            lines.append("")
        
        if self.errors_by_type:
            lines.append("Error Statistics:")
            for error_type, count in sorted(self.errors_by_type.items()):
                lines.append(f"  {error_type}: {count}")
            lines.append("")
        
        if self.user_approvals or self.user_rejections:
            lines.append("User Interaction Metrics:")
            lines.append(f"  Approvals: {self.user_approvals}")
            lines.append(f"  Rejections: {self.user_rejections}")
            lines.append(f"  Approval Rate: {self.get_approval_rate():.1f}%")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export metrics as JSON"""
        return json.dumps({
            "workflow_started": self.workflow_started,
            "workflow_completed": self.workflow_completed,
            "workflow_failed": self.workflow_failed,
            "success_rate": self.get_success_rate(),
            "avg_duration_ms": self.get_avg_duration() * 1000,
            "node_executions": dict(self.node_executions),
            "errors_by_type": dict(self.errors_by_type),
            "approval_rate": self.get_approval_rate()
        }, indent=2)

# Global metrics collector
metrics = Metrics()

# ==================== STRUCTURED LOGGER ====================

class StructuredLogger:
    """Logger with structured output"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
    
    def log_event(self, event_type: str, level: str = "INFO", **context):
        """Log structured event"""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            **context
        }
        
        log_level = getattr(logging, level.upper())
        self.logger.log(log_level, json.dumps(log_entry))

structured_logger = StructuredLogger(__name__)

# ==================== TRACING ====================

@dataclass
class Span:
    """Represents a trace span"""
    span_id: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    parent_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def finish(self):
        """Mark span as finished"""
        self.end_time = time.time()
    
    def duration_ms(self) -> float:
        """Get duration in milliseconds"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

class Tracer:
    """Simple distributed tracing"""
    
    def __init__(self):
        self.spans: List[Span] = []
        self.active_spans: Dict[str, Span] = {}
        self._span_counter = 0
    
    def start_span(self, operation: str, parent_id: Optional[str] = None, **metadata) -> str:
        """Start a new span"""
        self._span_counter += 1
        span_id = f"span-{self._span_counter}"
        
        span = Span(
            span_id=span_id,
            operation=operation,
            start_time=time.time(),
            parent_id=parent_id,
            metadata=metadata
        )
        
        self.spans.append(span)
        self.active_spans[span_id] = span
        
        logger.debug(f"Started span: {operation} ({span_id})")
        
        return span_id
    
    def finish_span(self, span_id: str):
        """Finish a span"""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.finish()
            del self.active_spans[span_id]
            
            logger.debug(f"Finished span: {span.operation} ({span.duration_ms():.1f}ms)")
    
    def get_trace_summary(self) -> str:
        """Get trace summary"""
        lines = ["TRACE SUMMARY", "=" * 60, ""]
        
        for span in self.spans:
            indent = "  " * (len([s for s in self.spans if s.span_id == span.parent_id]))
            duration = span.duration_ms()
            lines.append(f"{indent}{span.operation}: {duration:.1f}ms")
        
        return "\n".join(lines)

tracer = Tracer()

# ==================== MONITORED STATE ====================

class MonitoredState(TypedDict):
    """State for monitored workflow"""
    messages: Annotated[Sequence[BaseMessage], add]
    workflow_id: str
    task: str
    current_trace_id: str
    result: str

# ==================== MONITORED NODES ====================

def monitored_start(state: MonitoredState) -> dict:
    """Start with monitoring"""
    
    # Record metrics
    metrics.record_workflow_start()
    
    # Start trace
    trace_id = tracer.start_span("workflow", workflow_id=state["workflow_id"])
    
    # Structured log
    structured_logger.log_event(
        "workflow_started",
        level="INFO",
        workflow_id=state["workflow_id"],
        task=state["task"]
    )
    
    return {
        "current_trace_id": trace_id,
        "messages": [AIMessage(content="Workflow started with monitoring")]
    }

def monitored_process(state: MonitoredState) -> dict:
    """Processing node with monitoring"""
    
    node_name = "process"
    
    # Start span
    span_id = tracer.start_span(
        f"node_{node_name}",
        parent_id=state["current_trace_id"]
    )
    
    start_time = time.time()
    
    try:
        # Simulate processing
        time.sleep(0.5)
        
        result = f"Processed: {state['task']}"
        
        # Record success
        duration = time.time() - start_time
        metrics.record_node_execution(node_name, duration)
        
        structured_logger.log_event(
            "node_completed",
            level="INFO",
            node=node_name,
            duration_ms=duration * 1000
        )
        
        return {
            "result": result,
            "messages": [AIMessage(content=f"Node {node_name} complete")]
        }
    
    except Exception as e:
        # Record error
        metrics.record_error(type(e).__name__, node_name)
        
        structured_logger.log_event(
            "node_failed",
            level="ERROR",
            node=node_name,
            error=str(e)
        )
        
        raise
    
    finally:
        # Finish span
        tracer.finish_span(span_id)

def monitored_analyze(state: MonitoredState) -> dict:
    """Analysis node with monitoring"""
    
    node_name = "analyze"
    
    span_id = tracer.start_span(
        f"node_{node_name}",
        parent_id=state["current_trace_id"]
    )
    
    start_time = time.time()
    
    try:
        # Simulate analysis
        time.sleep(0.8)
        
        # Record metrics
        duration = time.time() - start_time
        metrics.record_node_execution(node_name, duration)
        
        return {
            "messages": [AIMessage(content="Analysis complete")]
        }
    
    finally:
        tracer.finish_span(span_id)

def monitored_complete(state: MonitoredState) -> dict:
    """Complete workflow with monitoring"""
    
    # Finish trace
    tracer.finish_span(state["current_trace_id"])
    
    # Calculate workflow duration
    workflow_span = next(
        (s for s in tracer.spans if s.span_id == state["current_trace_id"]),
        None
    )
    
    if workflow_span:
        duration = workflow_span.duration_ms() / 1000
        metrics.record_workflow_complete(duration)
    
    # Generate reports
    metrics_report = metrics.generate_report()
    trace_summary = tracer.get_trace_summary()
    
    result = f"""
MONITORED WORKFLOW COMPLETE
{'='*60}

{metrics_report}

{trace_summary}
"""
    
    structured_logger.log_event(
        "workflow_completed",
        level="INFO",
        workflow_id=state["workflow_id"],
        duration_ms=workflow_span.duration_ms() if workflow_span else 0
    )
    
    return {
        "messages": [AIMessage(content=result)]
    }

# ==================== BUILD MONITORED WORKFLOW ====================

monitored_workflow = StateGraph(MonitoredState)

monitored_workflow.add_node("start", monitored_start)
monitored_workflow.add_node("process", monitored_process)
monitored_workflow.add_node("analyze", monitored_analyze)
monitored_workflow.add_node("complete", monitored_complete)

monitored_workflow.set_entry_point("start")
monitored_workflow.add_edge("start", "process")
monitored_workflow.add_edge("process", "analyze")
monitored_workflow.add_edge("analyze", "complete")
monitored_workflow.add_edge("complete", END)

monitored_system = monitored_workflow.compile()

# ==================== DEMONSTRATION ====================

def demonstrate_monitoring():
    """Demonstrate monitoring and observability"""
    
    print("\n" + "="*60)
    print("MONITORING AND OBSERVABILITY DEMONSTRATION")
    print("="*60)
    
    # Reset metrics and traces
    global metrics, tracer
    metrics = Metrics()
    tracer = Tracer()
    
    # Run multiple workflows
    for i in range(3):
        print(f"\n--- Workflow {i+1} ---")
        
        state = {
            "messages": [HumanMessage(content="Start")],
            "workflow_id": f"wf-{i+1}",
            "task": f"Task {i+1}",
            "current_trace_id": "",
            "result": ""
        }
        
        result = monitored_system.invoke(state)
        
        time.sleep(0.5)  # Brief pause
    
    # Show final report
    print("\n" + "="*60)
    print("FINAL MONITORING REPORT")
    print("="*60)
    print(result["messages"][-1].content)
    
    # Export as JSON
    print("\n" + "="*60)
    print("METRICS (JSON Export)")
    print("="*60)
    print(metrics.to_json())

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    demonstrate_monitoring()
```

---

## 🚀 Part 4: Deployment Patterns

### Theory: Production Deployment

#### Deployment Models

**Model 1: Monolithic Deployment**
```
Single service containing entire workflow
Pros: Simple, easy to deploy
Cons: Scales as one unit, hard to update parts
```

**Model 2: Microservices**
```
Each agent/component as separate service
Pros: Independent scaling, easier updates
Cons: More complex, network overhead
```

**Model 3: Serverless**
```
Deploy as cloud functions (AWS Lambda, etc.)
Pros: Auto-scaling, pay per use
Cons: Cold starts, execution time limits
```

**Model 4: Hybrid**
```
Core workflow as service + serverless for spikes
Pros: Balance of control and flexibility
Cons: More complexity
```

#### Deployment Strategies

**Strategy 1: Blue-Green Deployment**
```
Blue (current): Running version 1
Green (new): Deploy version 2

Test Green → Switch traffic → Blue becomes backup
```

**Strategy 2: Rolling Deployment**
```
10 instances running version 1
Deploy version 2 to 2 instances → Test → Deploy to 2 more → Repeat
Gradual replacement
```

**Strategy 3: Canary Deployment**
```
Route 5% of traffic to new version
Monitor metrics → If good, increase to 25% → 50% → 100%
If bad, roll back immediately
```

**Strategy 4: Feature Flags**
```python
if feature_flag("new_algorithm_enabled"):
    result = new_algorithm()
else:
    result = old_algorithm()

# Can toggle without deployment
```

#### Container Deployment

**Dockerfile Example:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LANGCHAIN_TRACING_V2=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python", "main.py"]
```

**Docker Compose:**
```yaml
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://db:5432/agents
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    restart: unless-stopped
  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: agents
      POSTGRES_PASSWORD: secret
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

#### Kubernetes Deployment

**Basic Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langgraph-agent
  template:
    metadata:
      labels:
        app: langgraph-agent
    spec:
      containers:
      - name: agent
        image: myregistry/langgraph-agent:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

#### Environment Management

**Configuration Management:**
```python
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Environment
    environment: str = "development"  # development, staging, production
    
    # API Keys
    openai_api_key: str
    anthropic_api_key: str
    
    # Database
    database_url: str
    
    # Cache
    redis_url: Optional[str] = None
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    
    # Feature flags
    enable_caching: bool = True
    enable_tracing: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

**Environment-Specific Config:**
```
development:
  - Local database
  - Verbose logging
  - No rate limiting
  - Test API keys

staging:
  - Staging database
  - Normal logging
  - Relaxed rate limits
  - Test API keys

production:
  - Production database
  - Minimal logging
  - Strict rate limits
  - Production API keys
```

---

## 🧪 Part 5: Testing Strategies

### Theory: Testing Agent Systems

#### Testing Pyramid

```
        /\
       /  \      E2E Tests (Few, Slow, Expensive)
      /    \
     /------\    Integration Tests (Some, Medium)
    /        \
   /----------\  Unit Tests (Many, Fast, Cheap)
  /____________\
```

#### Types of Tests

**1. Unit Tests:**
```python
def test_node_function():
    """Test individual node"""
    state = {"input": "test"}
    result = my_node(state)
    assert result["output"] == "expected"
```

**2. Integration Tests:**
```python
def test_workflow_integration():
    """Test workflow with real components"""
    workflow = build_workflow()
    result = workflow.invoke({"input": "test"})
    assert result["status"] == "success"
```

**3. End-to-End Tests:**
```python
def test_full_system():
    """Test entire system including APIs"""
    response = requests.post("http://localhost:8000/workflow", json={
        "task": "test task"
    })
    assert response.status_code == 200
```

**4. Property-Based Tests:**
```python
from hypothesis import given, strategies as st

@given(st.text())
def test_node_handles_any_string(input_text):
    """Test with many random inputs"""
    state = {"input": input_text}
    result = my_node(state)
    # Should not crash
    assert "output" in result
```

#### Testing LLM-Based Systems

**Challenge:** Non-deterministic outputs

**Solutions:**

**1. Mock LLM Responses:**
```python
def test_with_mock_llm(mocker):
    """Test with mocked LLM"""
    mock_llm = mocker.patch("langchain_ollama.ChatOllama")
    mock_llm.invoke.return_value = "Expected response"
    
    result = agent_node(state)
    assert "Expected response" in result
```

**2. Test Behavior, Not Exact Output:**
```python
def test_sentiment_analysis():
    """Test behavior rather than exact output"""
    result = analyze_sentiment("I love this!")
    
    # Don't test exact wording
    # assert result == "The sentiment is positive."
    
    # Test that it's positive
    assert result["sentiment"] in ["positive", "very positive"]
    assert result["score"] > 0.5
```

**3. Use Consistent Seeds:**
```python
llm = ChatOllama(model="llama3.2", temperature=0, seed=42)
# More deterministic with temperature=0 and seed
```

**4. Regression Tests:**
```python
# Save known good outputs
golden_outputs = {
    "test_1": "Expected output 1",
    "test_2": "Expected output 2"
}

def test_regression():
    """Test that outputs haven't changed unexpectedly"""
    for test_input, expected in golden_outputs.items():
        result = agent(test_input)
        assert result == expected  # or use similarity threshold
```

#### Testing Human-in-Loop Systems

**Challenge:** Can't have human in automated tests

**Solution 1: Mock Human Decisions:**
```python
def test_approval_workflow_approved(mocker):
    """Test approval flow with mocked approval"""
    
    # Mock human decision
    mocker.patch("get_human_approval", return_value=True)
    
    result = approval_workflow.invoke(state)
    assert result["status"] == "approved"

def test_approval_workflow_rejected(mocker):
    """Test approval flow with mocked rejection"""
    
    mocker.patch("get_human_approval", return_value=False)
    
    result = approval_workflow.invoke(state)
    assert result["status"] == "rejected"
```

**Solution 2: Automated Approval in Test:**
```python
class TestApprovalHandler:
    """Auto-approve for testing"""
    def approve(self, request):
        return True  # Always approve in tests

workflow = build_workflow(
    approval_handler=TestApprovalHandler()
)
```

#### Load Testing

**Why:** Ensure system handles production load

**Tools:**
- Locust
- Apache JMeter
- k6

**Example Load Test:**
```python
from locust import HttpUser, task, between

class AgentUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def run_workflow(self):
        self.client.post("/workflow", json={
            "task": "Test task",
            "priority": "normal"
        })
```

**Run:**
```bash
# Simulate 100 users
locust -f load_test.py --users 100 --spawn-rate 10
```

---

### Implementation: Testing Framework

```python
import pytest
from unittest.mock import Mock, patch
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ==================== TEST FIXTURES ====================

@pytest.fixture
def sample_state():
    """Fixture for sample state"""
    return {
        "messages": [HumanMessage(content="Test input")],
        "task": "test task",
        "result": ""
    }

@pytest.fixture
def mock_llm(mocker):
    """Fixture for mocked LLM"""
    mock = mocker.patch("langchain_ollama.ChatOllama")
    mock.return_value.invoke.return_value = AIMessage(content="Mocked response")
    return mock

# ==================== UNIT TESTS ====================

class TestState(TypedDict):
    """State for testing"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    result: str
    count: int

def simple_node(state: TestState) -> dict:
    """Simple node for testing"""
    return {
        "result": f"Processed: {state['task']}",
        "count": 1
    }

def test_simple_node_function():
    """Test node function directly"""
    state = {
        "messages": [],
        "task": "test",
        "result": "",
        "count": 0
    }
    
    result = simple_node(state)
    
    assert result["result"] == "Processed: test"
    assert result["count"] == 1

def test_simple_node_with_fixture(sample_state):
    """Test using fixture"""
    result = simple_node(sample_state)
    
    assert "Processed:" in result["result"]
    assert result["count"] == 1

# ==================== INTEGRATION TESTS ====================

def test_simple_workflow():
    """Test complete workflow"""
    
    # Build workflow
    workflow = StateGraph(TestState)
    workflow.add_node("process", simple_node)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    
    graph = workflow.compile()
    
    # Execute
    initial_state = {
        "messages": [HumanMessage(content="Start")],
        "task": "integration test",
        "result": "",
        "count": 0
    }
    
    result = graph.invoke(initial_state)
    
    # Assertions
    assert result["result"] == "Processed: integration test"
    assert result["count"] == 1

def test_workflow_with_multiple_nodes():
    """Test workflow with multiple nodes"""
    
    def node_a(state: TestState) -> dict:
        return {"result": "A", "count": 1}
    
    def node_b(state: TestState) -> dict:
        return {"result": state["result"] + "B", "count": 1}
    
    # Build
    workflow = StateGraph(TestState)
    workflow.add_node("a", node_a)
    workflow.add_node("b", node_b)
    workflow.set_entry_point("a")
    workflow.add_edge("a", "b")
    workflow.add_edge("b", END)
    
    graph = workflow.compile()
    
    # Execute
    result = graph.invoke({
        "messages": [],
        "task": "test",
        "result": "",
        "count": 0
    })
    
    # Both nodes executed
    assert result["result"] == "AB"
    assert result["count"] == 2

# ==================== MOCKING TESTS ====================

def llm_based_node(state: TestState) -> dict:
    """Node that uses LLM"""
    from langchain_ollama import ChatOllama
    
    llm = ChatOllama(model="llama3.2")
    response = llm.invoke(state["task"])
    
    return {
        "result": response.content
    }

def test_llm_node_with_mock(mock_llm):
    """Test LLM-based node with mock"""
    
    state = {
        "messages": [],
        "task": "test task",
        "result": "",
        "count": 0
    }
    
    result = llm_based_node(state)
    
    # LLM was called
    mock_llm.return_value.invoke.assert_called_once()
    
    # Got mocked response
    assert result["result"] == "Mocked response"

# ==================== ERROR HANDLING TESTS ====================

def failing_node(state: TestState) -> dict:
    """Node that fails"""
    raise ValueError("Intentional error")

def test_error_handling():
    """Test that errors are handled"""
    
    state = {
        "messages": [],
        "task": "test",
        "result": "",
        "count": 0
    }
    
    with pytest.raises(ValueError, match="Intentional error"):
        failing_node(state)

def test_workflow_error_recovery():
    """Test workflow handles errors gracefully"""
    
    def safe_node(state: TestState) -> dict:
        try:
            return failing_node(state)
        except ValueError:
            return {"result": "Error handled", "count": 1}
    
    # Build workflow with error recovery
    workflow = StateGraph(TestState)
    workflow.add_node("safe", safe_node)
    workflow.set_entry_point("safe")
    workflow.add_edge("safe", END)
    
    graph = workflow.compile()
    
    result = graph.invoke({
        "messages": [],
        "task": "test",
        "result": "",
        "count": 0
    })
    
    assert result["result"] == "Error handled"

# ==================== PROPERTY-BASED TESTS ====================

try:
    from hypothesis import given, strategies as st
    
    @given(st.text())
    def test_node_handles_any_text(input_text):
        """Test node handles any string input"""
        state = {
            "messages": [],
            "task": input_text,
            "result": "",
            "count": 0
        }
        
        # Should not crash
        result = simple_node(state)
        
        # Should always produce result
        assert "result" in result
        assert result["count"] == 1

except ImportError:
    # Hypothesis not installed
    pass

# ==================== PERFORMANCE TESTS ====================

def test_node_performance():
    """Test node executes within time limit"""
    import time
    
    state = {
        "messages": [],
        "task": "performance test",
        "result": "",
        "count": 0
    }
    
    start = time.time()
    result = simple_node(state)
    duration = time.time() - start
    
    # Should complete in under 100ms
    assert duration < 0.1
    assert result["result"] == "Processed: performance test"

# ==================== SNAPSHOT TESTS ====================

def test_node_output_snapshot():
    """Test output matches snapshot"""
    
    state = {
        "messages": [],
        "task": "snapshot test",
        "result": "",
        "count": 0
    }
    
    result = simple_node(state)
    
    # Compare to known good output
    expected_snapshot = {
        "result": "Processed: snapshot test",
        "count": 1
    }
    
    assert result == expected_snapshot

# ==================== RUN TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 📈 Part 6: Scalability Considerations

### Theory: Scaling Agent Systems

#### Scalability Dimensions

**1. Vertical Scaling (Scale Up):**
```
Increase resources of single instance
- More CPU
- More RAM
- Faster storage

Pros: Simple
Cons: Limited, expensive at high end
```

**2. Horizontal Scaling (Scale Out):**
```
Add more instances
- Multiple servers
- Load balancing
- Distributed processing

Pros: Unlimited, cost-effective
Cons: Complexity, state management
```

#### Bottlenecks and Solutions

**Bottleneck 1: LLM API Rate Limits**
```
Problem: API providers limit requests/minute
Solutions:
- Request batching
- Multiple API keys/accounts
- Load distribution across providers
- Caching
```

**Bottleneck 2: Stateful Workflows**
```
Problem: State tied to specific instance
Solutions:
- Externalize state (Redis, PostgreSQL)
- Sticky sessions (same user → same instance)
- Stateless design where possible
```

**Bottleneck 3: Database Connections**
```
Problem: Limited concurrent connections
Solutions:
- Connection pooling
- Read replicas
- Caching layer (Redis)
- Database sharding
```

**Bottleneck 4: Memory Usage**
```
Problem: Large state objects in memory
Solutions:
- State compression
- Lazy loading
- Streaming processing
- Disk overflow
```

#### Horizontal Scaling Patterns

**Pattern 1: Stateless Workers**
```
┌─────────┐
│ Load    │
│ Balancer├─→ Worker 1
│         ├─→ Worker 2
└─────────┘  Worker 3

State stored in external DB
Any worker can handle any request
```

**Pattern 2: Partitioned Workflows**
```
User A workflows → Worker Group 1
User B workflows → Worker Group 2
User C workflows → Worker Group 3

Partition by user/tenant/region
```

**Pattern 3: Queue-Based**
```
Requests → Queue → Workers pull tasks
               ↓
          Processes asynchronously
          
Decouples submission from execution
Workers scale independently
```

#### Load Balancing Strategies

**Strategy 1: Round Robin**
```
Request 1 → Server A
Request 2 → Server B
Request 3 → Server C
Request 4 → Server A (cycle)
```

**Strategy 2: Least Connections**
```
Route to server with fewest active connections
Good for long-running requests
```

**Strategy 3: Weighted Distribution**
```
Server A (high spec): 50% traffic
Server B (medium): 30% traffic
Server C (low): 20% traffic
```

**Strategy 4: Geographic**
```
US users → US region servers
EU users → EU region servers
Reduces latency
```

#### Caching for Scale

**Cache What:**
```
✅ LLM responses (expensive, often repeated)
✅ Database queries (high frequency reads)
✅ External API calls (rate limited, slow)
✅ Computation results (expensive calculations)

❌ User-specific real-time data
❌ Critical transactional data
❌ Frequently changing data
```

**Cache Invalidation:**
```python
# Time-based (TTL)
cache.set(key, value, ttl=3600)  # 1 hour

# Event-based
on_data_update:
    cache.delete(key)

# Manual
cache.clear_pattern("user:*")
```

#### Async Processing

**Why Async:**
```
Synchronous:
User submits → Wait 30 seconds → Get result
(User blocked entire time)

Asynchronous:
User submits → Get job_id immediately
Poll /status/{job_id} or get webhook
(User not blocked)
```

**Implementation:**
```python
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def process_workflow(workflow_id):
    """Process workflow asynchronously"""
    result = workflow.invoke(initial_state)
    store_result(workflow_id, result)
    return result

# Submit
task = process_workflow.delay(workflow_id="wf-123")

# Check status
status = task.status  # PENDING, SUCCESS, FAILURE

# Get result
result = task.get()  # Blocks until complete
```

---

## 📚 Best Practices Summary

### Error Handling

**✅ DO:**
- Retry transient errors with exponential backoff
- Use circuit breakers for failing services
- Provide fallback mechanisms
- Log errors with full context
- Alert on critical errors

**❌ DON'T:**
- Retry without backoff
- Retry permanent errors
- Swallow errors silently
- Log sensitive data
- Alert on every minor issue

---

### Performance

**✅ DO:**
- Cache expensive operations
- Use connection pooling
- Batch when possible
- Monitor performance metrics
- Profile bottlenecks

**❌ DON'T:**
- Optimize prematurely
- Cache everything
- Ignore memory usage
- Skip performance testing
- Assume, measure instead

---

### Monitoring

**✅ DO:**
- Log structured events
- Track key metrics
- Set up alerting
- Use distributed tracing
- Monitor business metrics

**❌ DON'T:**
- Log unstructured text
- Monitor too many metrics
- Alert without actionability
- Forget to rotate logs
- Ignore alert fatigue

---

### Deployment

**✅ DO:**
- Use containerization
- Implement health checks
- Test in staging
- Deploy gradually
- Have rollback plan

**❌ DON'T:**
- Deploy directly to production
- Skip health checks
- Deploy Friday afternoon
- Forget database migrations
- Deploy without monitoring

---

### Testing

**✅ DO:**
- Write unit tests for nodes
- Test error handling
- Mock external dependencies
- Test edge cases
- Run load tests

**❌ DON'T:**
- Test only happy path
- Skip integration tests
- Test exact LLM outputs
- Ignore flaky tests
- Skip load testing

---

### Scalability

**✅ DO:**
- Design for horizontal scaling
- Externalize state
- Use caching
- Monitor resource usage
- Plan for growth

**❌ DON'T:**
- Assume vertical scaling works forever
- Keep state in memory only
- Ignore connection limits
- Wait for problems
- Over-engineer early

---

## ✅ Chapter 18 Complete!

**You now understand:**
- ✅ Error handling strategies (retry, circuit breaker, fallback)
- ✅ Performance optimization (caching, batching, profiling)
- ✅ Monitoring and observability (metrics, logs, traces)
- ✅ Deployment patterns (containers, K8s, strategies)
- ✅ Testing strategies (unit, integration, mocking)
- ✅ Scalability considerations (horizontal scaling, load balancing)
- ✅ Production best practices

**Key Takeaways:**
- Production systems need resilience (errors will happen)
- Performance requires measurement and optimization
- Observability enables debugging and improvement
- Deployment should be gradual and reversible
- Testing catches issues before production
- Scalability requires planning and architecture

---

## 🎓 Course Complete!

**Congratulations! You've completed the comprehensive LangGraph course.**

**You've learned:**
- Chapters 1-6: Core concepts, tools, state management
- Chapters 7-10: RAG, memory, routing, reasoning
- Chapters 11-13: Agent patterns, communication, cooperation
- Chapters 14-15: Roleplay/debate, game theory, competition
- Chapters 16-18: Advanced state, HITL, production engineering

**You're now equipped to build production-ready LangGraph agents!**

What would you like to explore next? Specific implementation questions? Deep dives on particular topics?
