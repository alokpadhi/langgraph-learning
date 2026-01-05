# Chapter 16: Advanced State Management

## 🔧 Introduction: Beyond Basic State

Up to this point, we've used LangGraph's state management with default behaviors:
- Simple TypedDict state schemas
- Default `add` operator for lists
- Basic state updates

**Chapter 16 goes deeper** into LangGraph's state management capabilities:
- Custom reducers to control how state merges
- Branching and parallel execution
- Nested workflows with sub-graphs
- Dynamic graph construction
- Streaming and async patterns

Think of this as **mastering the engine** - understanding how state flows through graphs and how to control that flow precisely.

---

## 🔄 Part 1: Custom Reducers and State Updates

### Theory: How State Updates Work

#### Default State Behavior

By default, when a node returns state updates:

```python
class State(TypedDict):
    messages: List[str]
    count: int

# Node returns
return {"count": 5}

# Result: count is REPLACED with 5
# Previous value is lost
```

**Default behavior = REPLACE**

#### The `Annotated` Type and Reducers

To change how state merges, use `Annotated` with a **reducer function**:

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[List[str], add]  # Use add reducer
    count: int  # No annotation = replace

# Node returns
return {"messages": ["new message"], "count": 5}

# Result:
# - messages: existing + ["new message"]  (accumulated)
# - count: 5 (replaced)
```

**Reducer** = function that determines how new values combine with existing values

#### Built-in Reducers

**1. `operator.add` - Concatenation/Addition**
```python
from operator import add

# For lists
messages: Annotated[List[str], add]
# [1, 2] + [3, 4] → [1, 2, 3, 4]

# For numbers
total: Annotated[int, add]
# 10 + 5 → 15

# For strings
text: Annotated[str, add]
# "Hello " + "World" → "Hello World"
```

**2. Custom Lambda Functions**
```python
# Take maximum
score: Annotated[int, lambda x, y: max(x, y)]
# 10, 15 → 15

# Take minimum
cost: Annotated[int, lambda x, y: min(x, y)]
# 100, 80 → 80

# Merge dictionaries
config: Annotated[dict, lambda x, y: {**x, **y}]
# {a: 1}, {b: 2} → {a: 1, b: 2}
```

**3. Custom Functions**
```python
def merge_unique(existing: List[str], new: List[str]) -> List[str]:
    """Merge lists keeping only unique items"""
    return list(set(existing + new))

tags: Annotated[List[str], merge_unique]
# ["a", "b"], ["b", "c"] → ["a", "b", "c"]
```

#### How Reducers Work Internally

**Step 1: Node Returns Update**
```python
def my_node(state: State) -> dict:
    return {"messages": ["new"], "count": 5}
```

**Step 2: LangGraph Applies Reducers**
```python
# For each field in return value:
for field, new_value in return_dict.items():
    if field has reducer:
        current_value = state[field]
        state[field] = reducer(current_value, new_value)
    else:
        state[field] = new_value  # Replace
```

**Step 3: State Updated**
```python
# Old state: {"messages": ["hello"], "count": 3}
# Node returns: {"messages": ["new"], "count": 5}
# New state: {"messages": ["hello", "new"], "count": 5}
```

#### When to Use Custom Reducers

✅ **Use Custom Reducers When:**
- Need to accumulate values (lists, counts, etc.)
- Want to merge rather than replace
- Need complex state combination logic
- Implementing custom state semantics

❌ **Don't Use When:**
- Simple replacement is sufficient
- State is independent between nodes
- Adds unnecessary complexity

#### Common Custom Reducer Patterns

**Pattern 1: Accumulate with Limit**
```python
def add_with_limit(existing: List, new: List, limit: int = 100) -> List:
    """Add to list but keep only last N items"""
    combined = existing + new
    return combined[-limit:]

messages: Annotated[List[str], lambda x, y: add_with_limit(x, y, 50)]
```

**Pattern 2: Merge Dictionaries Deeply**
```python
def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

config: Annotated[dict, deep_merge]
```

**Pattern 3: Append with Deduplication**
```python
def append_unique(existing: List[str], new: List[str]) -> List[str]:
    """Append only new unique items"""
    seen = set(existing)
    result = existing.copy()
    for item in new:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

tags: Annotated[List[str], append_unique]
```

**Pattern 4: Update with Validation**
```python
def update_if_valid(current: int, new: int) -> int:
    """Update only if new value is positive"""
    return new if new > 0 else current

score: Annotated[int, update_if_valid]
```

---

### Implementation: Custom Reducers System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CUSTOM REDUCERS ====================

# Custom reducer functions
def merge_unique_list(existing: List[str], new: List[str]) -> List[str]:
    """Merge lists keeping only unique items in order"""
    seen = set(existing)
    result = existing.copy()
    for item in new:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

def max_value(current: int, new: int) -> int:
    """Keep maximum value"""
    return max(current, new)

def min_value(current: float, new: float) -> float:
    """Keep minimum value"""
    return min(current, new)

def deep_merge_dict(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result

def accumulate_with_limit(existing: List[str], new: List[str], limit: int = 10) -> List[str]:
    """Accumulate but keep only last N items"""
    combined = existing + new
    return combined[-limit:]

# Wrapper for limit
def limited_accumulator(limit: int):
    """Create accumulator with specific limit"""
    def accumulator(existing: List[str], new: List[str]) -> List[str]:
        return accumulate_with_limit(existing, new, limit)
    return accumulator

# ==================== STATE WITH CUSTOM REDUCERS ====================

class AdvancedState(TypedDict):
    """State demonstrating various reducers"""
    # Standard add - concatenates
    messages: Annotated[Sequence[BaseMessage], add]
    
    # Custom - unique tags only
    tags: Annotated[List[str], merge_unique_list]
    
    # Custom - running sum
    total_cost: Annotated[float, add]
    
    # Custom - keep maximum
    max_score: Annotated[int, max_value]
    
    # Custom - keep minimum
    min_latency: Annotated[float, min_value]
    
    # Custom - deep merge
    config: Annotated[Dict[str, Any], deep_merge_dict]
    
    # Limited accumulator - keep last 5
    recent_events: Annotated[List[str], limited_accumulator(5)]
    
    # No annotation - replace
    current_status: str
    iteration: int

llm = ChatOllama(model="llama3.2", temperature=0.3)

# ==================== DEMO NODES ====================

def node_a(state: AdvancedState) -> dict:
    """First node - adds various state updates"""
    logger.info("Node A: Adding state updates")
    
    return {
        "messages": [AIMessage(content="Node A processed")],
        "tags": ["processing", "node_a"],
        "total_cost": 10.5,
        "max_score": 85,
        "min_latency": 120.5,
        "config": {"settings": {"timeout": 30}},
        "recent_events": ["node_a_started", "node_a_completed"],
        "current_status": "processing",
        "iteration": 1
    }

def node_b(state: AdvancedState) -> dict:
    """Second node - adds more updates"""
    logger.info("Node B: Adding state updates")
    
    return {
        "messages": [AIMessage(content="Node B processed")],
        "tags": ["node_b", "processing"],  # "processing" is duplicate
        "total_cost": 15.3,
        "max_score": 92,  # Higher - should replace max_score
        "min_latency": 115.2,  # Lower - should replace min_latency
        "config": {"settings": {"retries": 3}, "features": {"logging": True}},
        "recent_events": ["node_b_started", "node_b_validated", "node_b_completed"],
        "current_status": "validation",
        "iteration": 2
    }

def node_c(state: AdvancedState) -> dict:
    """Third node - adds final updates"""
    logger.info("Node C: Adding state updates")
    
    return {
        "messages": [AIMessage(content="Node C processed")],
        "tags": ["node_c", "complete", "node_a"],  # "node_a" is duplicate
        "total_cost": 8.7,
        "max_score": 88,  # Lower than 92 - should NOT replace
        "min_latency": 125.0,  # Higher than 115.2 - should NOT replace
        "config": {"settings": {"timeout": 60}, "auth": {"enabled": True}},
        "recent_events": ["node_c_started", "node_c_finalized"],
        "current_status": "complete",
        "iteration": 3
    }

def display_state(state: AdvancedState) -> dict:
    """Display final state"""
    logger.info("Displaying final state")
    
    result = f"""CUSTOM REDUCERS DEMONSTRATION

Final State After All Nodes:

1. Messages (reducer: add):
{chr(10).join([f"   - {msg.content}" for msg in state['messages']])}

2. Tags (reducer: merge_unique_list):
   {state['tags']}
   → Note: "processing" and "node_a" appear only once despite multiple additions

3. Total Cost (reducer: add):
   ${state['total_cost']:.2f}
   → Sum: $10.5 + $15.3 + $8.7 = $34.5

4. Max Score (reducer: max_value):
   {state['max_score']}
   → Kept highest: max(85, 92, 88) = 92

5. Min Latency (reducer: min_value):
   {state['min_latency']:.1f}ms
   → Kept lowest: min(120.5, 115.2, 125.0) = 115.2

6. Config (reducer: deep_merge_dict):
   {state['config']}
   → Settings merged: timeout updated to 60, retries added, auth added

7. Recent Events (reducer: limited_accumulator(5)):
   {state['recent_events']}
   → Kept last 5 only (from total of 7 events)

8. Current Status (no reducer - replace):
   "{state['current_status']}"
   → Simply replaced: "processing" → "validation" → "complete"

9. Iteration (no reducer - replace):
   {state['iteration']}
   → Simply replaced: 1 → 2 → 3"""
    
    return {
        "messages": [AIMessage(content=result)]
    }

# Build workflow
reducer_workflow = StateGraph(AdvancedState)

reducer_workflow.add_node("node_a", node_a)
reducer_workflow.add_node("node_b", node_b)
reducer_workflow.add_node("node_c", node_c)
reducer_workflow.add_node("display", display_state)

reducer_workflow.set_entry_point("node_a")
reducer_workflow.add_edge("node_a", "node_b")
reducer_workflow.add_edge("node_b", "node_c")
reducer_workflow.add_edge("node_c", "display")
reducer_workflow.add_edge("display", END)

reducer_system = reducer_workflow.compile()

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CUSTOM REDUCERS DEMONSTRATION")
    print("="*60)
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content="Starting workflow")],
        "tags": [],
        "total_cost": 0.0,
        "max_score": 0,
        "min_latency": float('inf'),
        "config": {},
        "recent_events": [],
        "current_status": "initialized",
        "iteration": 0
    }
    
    result = reducer_system.invoke(initial_state)
    
    print(f"\n{result['messages'][-1].content}")
```

---

## 🌳 Part 2: Branching and Parallel Execution

### Theory: Branching in Graphs

#### What is Branching?

**Branching** means the graph's execution path splits based on conditions, allowing:
- Conditional execution (if-then logic)
- Multiple paths through the graph
- Dynamic routing

```
         ┌──────────┐
         │  Start   │
         └────┬─────┘
              │
         ┌────▼─────┐
         │ Classify │
         └────┬─────┘
              │
        ┌─────┴─────┐
        │           │
    ┌───▼───┐   ┌───▼────┐
    │Path A │   │ Path B │
    └───┬───┘   └───┬────┘
        │           │
        └─────┬─────┘
              │
         ┌────▼─────┐
         │   End    │
         └──────────┘
```

#### Types of Branching

**1. Conditional Edges**
```python
def route_based_on_condition(state: State) -> str:
    if state["value"] > 10:
        return "path_a"
    else:
        return "path_b"

workflow.add_conditional_edges(
    "classifier",
    route_based_on_condition,
    {
        "path_a": "node_a",
        "path_b": "node_b"
    }
)
```

**2. Multi-way Branching**
```python
def route_to_multiple(state: State) -> str:
    category = state["category"]
    return category  # Returns "tech", "finance", or "health"

workflow.add_conditional_edges(
    "router",
    route_to_multiple,
    {
        "tech": "tech_handler",
        "finance": "finance_handler",
        "health": "health_handler"
    }
)
```

**3. Dynamic Branching**
```python
def route_dynamically(state: State) -> str:
    # Route based on runtime conditions
    if state["urgent"]:
        return "fast_path"
    elif state["complexity"] == "high":
        return "expert_path"
    else:
        return "standard_path"
```

#### What is Parallel Execution?

**Parallel execution** means multiple nodes run simultaneously:

```
         ┌──────────┐
         │  Start   │
         └────┬─────┘
              │
         ┌────▼─────┐
         │ Dispatch │
         └────┬─────┘
              │
        ┌─────┼─────┐
        │     │     │
    ┌───▼──┐ │ ┌───▼──┐
    │Node A│ │ │Node B│
    └───┬──┘ │ └───┬──┘
        │    │     │
        │ ┌──▼──┐  │
        │ │Node │  │
        │ │  C  │  │
        │ └──┬──┘  │
        │    │     │
        └────┼─────┘
             │
        ┌────▼─────┐
        │  Merge   │
        └──────────┘
```

**All paths execute concurrently**, then results merge.

#### Implementing Parallel Execution

**Method 1: Multiple Edges from Same Node**
```python
workflow.add_node("dispatcher", dispatch)
workflow.add_node("worker_a", worker_a)
workflow.add_node("worker_b", worker_b)
workflow.add_node("worker_c", worker_c)
workflow.add_node("merger", merge_results)

# Parallel edges
workflow.add_edge("dispatcher", "worker_a")
workflow.add_edge("dispatcher", "worker_b")
workflow.add_edge("dispatcher", "worker_c")

# All converge to merger
workflow.add_edge("worker_a", "merger")
workflow.add_edge("worker_b", "merger")
workflow.add_edge("worker_c", "merger")
```

**When dispatcher completes:**
- worker_a, worker_b, worker_c ALL start immediately
- Run in parallel
- merger waits for ALL to complete

#### Parallel vs Sequential Trade-offs

| Aspect | Sequential | Parallel |
|--------|-----------|----------|
| **Speed** | Slower (one at a time) | Faster (concurrent) |
| **Dependencies** | Easy (linear order) | Complex (must handle) |
| **State Management** | Simple | Must merge carefully |
| **Debugging** | Easier | Harder (race conditions) |
| **Resource Use** | Lower | Higher |

#### Parallel Execution Patterns

**Pattern 1: Fan-Out, Fan-In**
```
One node dispatches to multiple workers
Workers process independently
Results merge back together
```

**Pattern 2: Pipeline with Parallel Stages**
```
Stage 1 → [Stage 2A, Stage 2B, Stage 2C] → Stage 3
Some stages have parallel sub-tasks
```

**Pattern 3: Independent Parallel Tracks**
```
Split into completely independent paths
Each path has multiple steps
Converge at end
```

#### When to Use Branching vs Parallel

**Use Branching When:**
- Need different logic for different conditions
- Mutually exclusive paths
- One path is sufficient

**Use Parallel When:**
- Tasks are independent
- Want faster execution
- Can merge results
- No dependencies between tasks

---

### Implementation: Branching and Parallel System

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== BRANCHING EXAMPLE ====================

class BranchingState(TypedDict):
    """State for branching demo"""
    messages: Annotated[Sequence[BaseMessage], add]
    task_type: str
    complexity: str
    path_taken: Annotated[List[str], add]

def classify_task(state: BranchingState) -> dict:
    """Classify the task"""
    logger.info("Classifying task")
    
    task = state["messages"][-1].content
    
    # Simple classification
    if "urgent" in task.lower():
        task_type = "urgent"
    elif "analysis" in task.lower():
        task_type = "analysis"
    else:
        task_type = "standard"
    
    if len(task) > 50:
        complexity = "high"
    else:
        complexity = "low"
    
    logger.info(f"Classified: type={task_type}, complexity={complexity}")
    
    return {
        "task_type": task_type,
        "complexity": complexity,
        "path_taken": ["classifier"]
    }

def route_by_type(state: BranchingState) -> str:
    """Route based on task type"""
    return state["task_type"]

def urgent_handler(state: BranchingState) -> dict:
    """Handle urgent tasks"""
    logger.info("🔴 URGENT handler")
    time.sleep(0.5)
    
    return {
        "messages": [AIMessage(content="Processed as URGENT with priority")],
        "path_taken": ["urgent_handler"]
    }

def analysis_handler(state: BranchingState) -> dict:
    """Handle analysis tasks"""
    logger.info("📊 ANALYSIS handler")
    time.sleep(0.5)
    
    return {
        "messages": [AIMessage(content="Processed as ANALYSIS with deep review")],
        "path_taken": ["analysis_handler"]
    }

def standard_handler(state: BranchingState) -> dict:
    """Handle standard tasks"""
    logger.info("📋 STANDARD handler")
    time.sleep(0.5)
    
    return {
        "messages": [AIMessage(content="Processed as STANDARD task")],
        "path_taken": ["standard_handler"]
    }

def finalize_branching(state: BranchingState) -> dict:
    """Finalize after branching"""
    
    path_str = " → ".join(state["path_taken"])
    
    result = f"""BRANCHING EXAMPLE RESULT

Task Type: {state['task_type']}
Complexity: {state['complexity']}
Path Taken: {path_str}

Processing: {state['messages'][-1].content}"""
    
    return {
        "messages": [AIMessage(content=result)],
        "path_taken": ["finalizer"]
    }

# Build branching workflow
branching_workflow = StateGraph(BranchingState)

branching_workflow.add_node("classifier", classify_task)
branching_workflow.add_node("urgent", urgent_handler)
branching_workflow.add_node("analysis", analysis_handler)
branching_workflow.add_node("standard", standard_handler)
branching_workflow.add_node("finalize", finalize_branching)

branching_workflow.set_entry_point("classifier")

# Conditional branching
branching_workflow.add_conditional_edges(
    "classifier",
    route_by_type,
    {
        "urgent": "urgent",
        "analysis": "analysis",
        "standard": "standard"
    }
)

# All paths converge to finalize
branching_workflow.add_edge("urgent", "finalize")
branching_workflow.add_edge("analysis", "finalize")
branching_workflow.add_edge("standard", "finalize")
branching_workflow.add_edge("finalize", END)

branching_system = branching_workflow.compile()

# ==================== PARALLEL EXECUTION EXAMPLE ====================

class ParallelState(TypedDict):
    """State for parallel demo"""
    messages: Annotated[Sequence[BaseMessage], add]
    task: str
    worker_results: Annotated[List[str], add]
    start_time: float
    end_time: float

def dispatch_parallel(state: ParallelState) -> dict:
    """Dispatch task to parallel workers"""
    logger.info("📤 Dispatching to parallel workers")
    
    return {
        "start_time": time.time(),
        "messages": [AIMessage(content="Dispatched to parallel workers")]
    }

def worker_alpha(state: ParallelState) -> dict:
    """Worker Alpha"""
    logger.info("⚙️ Worker Alpha: Processing")
    time.sleep(1.0)  # Simulate work
    
    return {
        "worker_results": ["Worker Alpha completed analysis"],
        "messages": [AIMessage(content="Worker Alpha done")]
    }

def worker_beta(state: ParallelState) -> dict:
    """Worker Beta"""
    logger.info("⚙️ Worker Beta: Processing")
    time.sleep(1.2)  # Simulate work
    
    return {
        "worker_results": ["Worker Beta completed validation"],
        "messages": [AIMessage(content="Worker Beta done")]
    }

def worker_gamma(state: ParallelState) -> dict:
    """Worker Gamma"""
    logger.info("⚙️ Worker Gamma: Processing")
    time.sleep(0.8)  # Simulate work
    
    return {
        "worker_results": ["Worker Gamma completed enrichment"],
        "messages": [AIMessage(content="Worker Gamma done")]
    }

def merge_parallel_results(state: ParallelState) -> dict:
    """Merge results from parallel workers"""
    logger.info("🔀 Merging parallel results")
    
    end_time = time.time()
    duration = end_time - state["start_time"]
    
    results_text = "\n".join([f"  - {r}" for r in state["worker_results"]])
    
    result = f"""PARALLEL EXECUTION RESULT

Task: {state['task']}

Workers: 3 (Alpha, Beta, Gamma)
Execution: Parallel

Results:
{results_text}

Total Time: {duration:.2f} seconds
→ All workers ran simultaneously

Sequential would have taken: ~3.0 seconds (1.0 + 1.2 + 0.8)
Parallel took: ~{duration:.2f} seconds (max of individual times)
Speed improvement: {3.0/duration:.1f}x"""
    
    return {
        "end_time": end_time,
        "messages": [AIMessage(content=result)]
    }

# Build parallel workflow
parallel_workflow = StateGraph(ParallelState)

parallel_workflow.add_node("dispatch", dispatch_parallel)
parallel_workflow.add_node("worker_alpha", worker_alpha)
parallel_workflow.add_node("worker_beta", worker_beta)
parallel_workflow.add_node("worker_gamma", worker_gamma)
parallel_workflow.add_node("merge", merge_parallel_results)

parallel_workflow.set_entry_point("dispatch")

# Parallel edges - all three workers start after dispatch
parallel_workflow.add_edge("dispatch", "worker_alpha")
parallel_workflow.add_edge("dispatch", "worker_beta")
parallel_workflow.add_edge("dispatch", "worker_gamma")

# All workers converge to merge
parallel_workflow.add_edge("worker_alpha", "merge")
parallel_workflow.add_edge("worker_beta", "merge")
parallel_workflow.add_edge("worker_gamma", "merge")

parallel_workflow.add_edge("merge", END)

parallel_system = parallel_workflow.compile()

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BRANCHING EXAMPLE")
    print("="*60)
    
    # Test branching with different task types
    tasks = [
        "Please handle this urgent issue immediately",
        "Run detailed analysis on the quarterly data",
        "Process standard request"
    ]
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        
        result = branching_system.invoke({
            "messages": [HumanMessage(content=task)],
            "task_type": "",
            "complexity": "",
            "path_taken": []
        })
        
        print(f"\n{result['messages'][-1].content}")
    
    # Test parallel execution
    print(f"\n{'='*60}")
    print("PARALLEL EXECUTION EXAMPLE")
    print(f"{'='*60}")
    
    result = parallel_system.invoke({
        "messages": [HumanMessage(content="Process comprehensive report")],
        "task": "Comprehensive report processing",
        "worker_results": [],
        "start_time": 0.0,
        "end_time": 0.0
    })
    
    print(f"\n{result['messages'][-1].content}")
```

---

## 🪆 Part 3: Sub-Graphs and Nested Workflows

### Theory: Sub-Graphs

#### What Are Sub-Graphs?

**Sub-graphs** are compiled LangGraph workflows used as nodes within a parent workflow. This enables:
- **Modularity**: Encapsulate complex logic
- **Reusability**: Use same sub-graph in multiple places
- **Composition**: Build complex systems from simpler parts
- **Abstraction**: Hide implementation details

```
Parent Graph:
┌────────────────────────────────────┐
│  ┌─────┐   ┌─────────┐   ┌─────┐ │
│  │Node1│──→│Sub-Graph│──→│Node3│ │
│  └─────┘   └────┬────┘   └─────┘ │
└───────────────────┼────────────────┘
                    │
        Sub-Graph (complete workflow):
        ┌───────────┼───────────┐
        │  ┌─────┐  │  ┌─────┐  │
        │  │  A  │──┼─→│  B  │  │
        │  └─────┘  │  └──┬──┘  │
        │           │     │     │
        │           │  ┌──▼──┐  │
        │           │  │  C  │  │
        │           │  └─────┘  │
        └───────────────────────┘
```

#### Why Use Sub-Graphs?

**1. Manage Complexity:**
```
Instead of one giant graph with 50 nodes
→ Break into logical sub-graphs of 5-10 nodes each
```

**2. Team Collaboration:**
```
Team A: Develops data processing sub-graph
Team B: Develops analysis sub-graph
Team C: Integrates both in parent graph
```

**3. Testing:**
```
Test sub-graphs independently
Then test parent graph
Easier than testing monolithic system
```

**4. Reusability:**
```
validation_subgraph used in:
- User input validation
- API response validation
- Data import validation
```

#### Creating Sub-Graphs

**Step 1: Define Sub-Graph State**
```python
class SubGraphState(TypedDict):
    """State for sub-graph"""
    input_data: str
    processed_data: str
    validation_passed: bool
```

**Step 2: Build Sub-Graph**
```python
subgraph = StateGraph(SubGraphState)
subgraph.add_node("process", process_node)
subgraph.add_node("validate", validate_node)
subgraph.set_entry_point("process")
subgraph.add_edge("process", "validate")
subgraph.add_edge("validate", END)

# Compile sub-graph
compiled_subgraph = subgraph.compile()
```

**Step 3: Use as Node in Parent**
```python
def invoke_subgraph_node(state: ParentState) -> dict:
    """Wrapper to invoke sub-graph"""
    
    # Prepare input for sub-graph
    subgraph_input = {
        "input_data": state["data"],
        "processed_data": "",
        "validation_passed": False
    }
    
    # Invoke sub-graph
    result = compiled_subgraph.invoke(subgraph_input)
    
    # Extract relevant output for parent
    return {
        "data": result["processed_data"],
        "valid": result["validation_passed"]
    }

# Add to parent
parent_workflow.add_node("subgraph", invoke_subgraph_node)
```

#### State Mapping Between Parent and Sub-Graph

**Problem:** Parent and sub-graph have different state schemas.

**Solution:** Explicit mapping in wrapper node.

```python
# Parent state
class ParentState(TypedDict):
    user_input: str
    final_output: str
    success: bool

# Sub-graph state  
class SubGraphState(TypedDict):
    text: str
    processed_text: str
    error: str

# Mapping
def subgraph_wrapper(state: ParentState) -> dict:
    # Map parent → sub-graph
    subgraph_input = {
        "text": state["user_input"],
        "processed_text": "",
        "error": ""
    }
    
    result = subgraph.invoke(subgraph_input)
    
    # Map sub-graph → parent
    return {
        "final_output": result["processed_text"],
        "success": not bool(result["error"])
    }
```

#### Nested Sub-Graphs

Sub-graphs can contain other sub-graphs:

```
Level 1 (Parent):
  ├─ Node A
  ├─ Sub-Graph B (Level 2)
  │   ├─ Node B1
  │   ├─ Sub-Graph B2 (Level 3)
  │   │   ├─ Node B2a
  │   │   └─ Node B2b
  │   └─ Node B3
  └─ Node C
```

**Use cases:**
- Very complex systems
- Multiple levels of abstraction
- Hierarchical organization

**Warning:** Don't over-nest! 2-3 levels max for maintainability.

---

### Implementation: Sub-Graphs System

```python
from typing import TypedDict, Annotated, Sequence, List
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== SUB-GRAPH 1: DATA PROCESSOR ====================

class DataProcessorState(TypedDict):
    """State for data processing sub-graph"""
    messages: Annotated[Sequence[BaseMessage], add]
    raw_data: str
    cleaned_data: str
    validated: bool

def clean_data(state: DataProcessorState) -> dict:
    """Clean the data"""
    logger.info("  [SUB] Cleaning data")
    
    raw = state["raw_data"]
    cleaned = raw.strip().lower().replace("  ", " ")
    
    return {
        "cleaned_data": cleaned,
        "messages": [AIMessage(content="Data cleaned")]
    }

def validate_data(state: DataProcessorState) -> dict:
    """Validate the data"""
    logger.info("  [SUB] Validating data")
    
    cleaned = state["cleaned_data"]
    validated = len(cleaned) > 0 and len(cleaned) < 1000
    
    return {
        "validated": validated,
        "messages": [AIMessage(content=f"Data valid: {validated}")]
    }

# Build data processor sub-graph
data_processor = StateGraph(DataProcessorState)
data_processor.add_node("clean", clean_data)
data_processor.add_node("validate", validate_data)
data_processor.set_entry_point("clean")
data_processor.add_edge("clean", "validate")
data_processor.add_edge("validate", END)

# Compile sub-graph
data_processor_compiled = data_processor.compile()
logger.info("✅ Data Processor sub-graph compiled")

# ==================== SUB-GRAPH 2: ANALYZER ====================

class AnalyzerState(TypedDict):
    """State for analyzer sub-graph"""
    messages: Annotated[Sequence[BaseMessage], add]
    input_text: str
    word_count: int
    sentiment: str
    key_topics: List[str]

def count_words(state: AnalyzerState) -> dict:
    """Count words"""
    logger.info("  [SUB] Counting words")
    
    words = state["input_text"].split()
    
    return {
        "word_count": len(words),
        "messages": [AIMessage(content=f"Word count: {len(words)}")]
    }

def analyze_sentiment(state: AnalyzerState) -> dict:
    """Simple sentiment analysis"""
    logger.info("  [SUB] Analyzing sentiment")
    
    text = state["input_text"].lower()
    
    # Simple keyword-based sentiment
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "hate", "sad"]
    
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    if pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > pos_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "messages": [AIMessage(content=f"Sentiment: {sentiment}")]
    }

def extract_topics(state: AnalyzerState) -> dict:
    """Extract key topics"""
    logger.info("  [SUB] Extracting topics")
    
    # Simple: just take nouns (words longer than 4 chars)
    words = state["input_text"].split()
    topics = [w for w in words if len(w) > 4][:3]
    
    return {
        "key_topics": topics,
        "messages": [AIMessage(content=f"Topics: {topics}")]
    }

# Build analyzer sub-graph
analyzer = StateGraph(AnalyzerState)
analyzer.add_node("count", count_words)
analyzer.add_node("sentiment", analyze_sentiment)
analyzer.add_node("topics", extract_topics)
analyzer.set_entry_point("count")
analyzer.add_edge("count", "sentiment")
analyzer.add_edge("sentiment", "topics")
analyzer.add_edge("topics", END)

# Compile sub-graph
analyzer_compiled = analyzer.compile()
logger.info("✅ Analyzer sub-graph compiled")

# ==================== PARENT GRAPH ====================

class ParentState(TypedDict):
    """State for parent workflow"""
    messages: Annotated[Sequence[BaseMessage], add]
    user_input: str
    processing_status: str
    analysis_summary: str

def receive_input(state: ParentState) -> dict:
    """Receive user input"""
    logger.info("[PARENT] Receiving input")
    
    return {
        "messages": [AIMessage(content="Input received")],
        "processing_status": "received"
    }

def invoke_data_processor(state: ParentState) -> dict:
    """Invoke data processor sub-graph"""
    logger.info("[PARENT] Invoking Data Processor sub-graph")
    
    # Prepare input for sub-graph
    subgraph_input = {
        "messages": [],
        "raw_data": state["user_input"],
        "cleaned_data": "",
        "validated": False
    }
    
    # Invoke sub-graph
    result = data_processor_compiled.invoke(subgraph_input)
    
    # Extract results
    cleaned = result["cleaned_data"]
    valid = result["validated"]
    
    logger.info(f"[PARENT] Data processor returned: validated={valid}")
    
    return {
        "user_input": cleaned,  # Update with cleaned version
        "processing_status": "cleaned and validated" if valid else "validation failed",
        "messages": [AIMessage(content=f"Data processing complete (valid: {valid})")]
    }

def invoke_analyzer(state: ParentState) -> dict:
    """Invoke analyzer sub-graph"""
    logger.info("[PARENT] Invoking Analyzer sub-graph")
    
    # Prepare input for sub-graph
    subgraph_input = {
        "messages": [],
        "input_text": state["user_input"],
        "word_count": 0,
        "sentiment": "",
        "key_topics": []
    }
    
    # Invoke sub-graph
    result = analyzer_compiled.invoke(subgraph_input)
    
    # Extract results
    word_count = result["word_count"]
    sentiment = result["sentiment"]
    topics = result["key_topics"]
    
    logger.info(f"[PARENT] Analyzer returned: {word_count} words, {sentiment} sentiment")
    
    summary = f"Words: {word_count}, Sentiment: {sentiment}, Topics: {', '.join(topics)}"
    
    return {
        "analysis_summary": summary,
        "processing_status": "analysis complete",
        "messages": [AIMessage(content=f"Analysis complete: {summary}")]
    }

def finalize_parent(state: ParentState) -> dict:
    """Finalize parent workflow"""
    logger.info("[PARENT] Finalizing")
    
    result = f"""SUB-GRAPH WORKFLOW RESULT

Original Input: {state['user_input']}

Processing Steps:
1. Data Processor Sub-Graph:
   - Cleaned and validated data
   - Status: {state['processing_status']}

2. Analyzer Sub-Graph:
   - Performed comprehensive analysis
   - Results: {state['analysis_summary']}

Final Status: {state['processing_status']}"""
    
    return {
        "messages": [AIMessage(content=result)]
    }

# Build parent workflow
parent_workflow = StateGraph(ParentState)

parent_workflow.add_node("receive", receive_input)
parent_workflow.add_node("process_data", invoke_data_processor)
parent_workflow.add_node("analyze", invoke_analyzer)
parent_workflow.add_node("finalize", finalize_parent)

parent_workflow.set_entry_point("receive")
parent_workflow.add_edge("receive", "process_data")
parent_workflow.add_edge("process_data", "analyze")
parent_workflow.add_edge("analyze", "finalize")
parent_workflow.add_edge("finalize", END)

parent_system = parent_workflow.compile()
logger.info("✅ Parent workflow compiled")

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SUB-GRAPHS AND NESTED WORKFLOWS")
    print("="*60)
    
    test_input = "This is a GREAT example  of data that needs processing and analysis!"
    
    print(f"\nInput: {test_input}")
    print("\nExecuting parent workflow with two sub-graphs...")
    print("-" * 60)
    
    result = parent_system.invoke({
        "messages": [HumanMessage(content="Start processing")],
        "user_input": test_input,
        "processing_status": "",
        "analysis_summary": ""
    })
    
    print(f"\n{result['messages'][-1].content}")
```

---

## 🎨 Part 4: Dynamic Graph Construction

### Theory: Dynamic Graphs

#### What Are Dynamic Graphs?

**Dynamic graphs** are workflows where the structure (nodes, edges) is determined **at runtime** rather than being fixed at compile time.

```
Static Graph (Fixed):
┌─────┐   ┌─────┐   ┌─────┐
│  A  │──→│  B  │──→│  C  │
└─────┘   └─────┘   └─────┘
Always the same structure

Dynamic Graph (Runtime-determined):
User request → Analyze → Build graph for this specific request
Could be: A → B → C
       or: A → D → E → F
       or: A → B
```

#### Why Dynamic Graphs?

**1. Adapt to Input:**
```
Different inputs need different processing:
- Simple query: 2 steps
- Complex query: 10 steps
- Research task: Custom pipeline
```

**2. User Configuration:**
```
User specifies workflow:
"First validate, then enrich, then analyze, skip visualization"
→ Build graph matching these requirements
```

**3. Conditional Complexity:**
```
Based on runtime conditions:
- If data is small: simple processing
- If data is large: distributed processing with checkpoints
```

**4. Plugin Architecture:**
```
Load plugins dynamically
Build graph incorporating available plugins
Extensible system
```

#### Approaches to Dynamic Graphs

**Approach 1: Conditional Structure**
```python
def build_graph_for_task(task_type: str):
    """Build different graph based on task type"""
    
    workflow = StateGraph(State)
    workflow.add_node("start", start_node)
    
    if task_type == "simple":
        workflow.add_node("process", simple_processor)
    elif task_type == "complex":
        workflow.add_node("analyze", analyzer)
        workflow.add_node("enrich", enricher)
        workflow.add_node("process", complex_processor)
        workflow.add_edge("analyze", "enrich")
        workflow.add_edge("enrich", "process")
    
    # Continue building...
    return workflow.compile()
```

**Approach 2: Pipeline Builder**
```python
def build_pipeline(steps: List[str]):
    """Build graph from list of step names"""
    
    workflow = StateGraph(State)
    
    # Map step names to node functions
    step_functions = {
        "validate": validate_node,
        "clean": clean_node,
        "enrich": enrich_node,
        "analyze": analyze_node
    }
    
    # Add nodes
    for step in steps:
        workflow.add_node(step, step_functions[step])
    
    # Connect sequentially
    workflow.set_entry_point(steps[0])
    for i in range(len(steps) - 1):
        workflow.add_edge(steps[i], steps[i + 1])
    workflow.add_edge(steps[-1], END)
    
    return workflow.compile()
```

**Approach 3: Graph Templates**
```python
# Pre-defined templates
TEMPLATES = {
    "data_pipeline": ["ingest", "validate", "transform", "load"],
    "ml_workflow": ["prepare", "train", "evaluate", "deploy"],
    "etl": ["extract", "transform", "load"]
}

def build_from_template(template_name: str, customizations: dict):
    """Build graph from template with customizations"""
    
    base_steps = TEMPLATES[template_name]
    
    # Apply customizations
    if customizations.get("include_validation"):
        base_steps.insert(1, "validate")
    
    if customizations.get("parallel_transform"):
        # Add parallel branches
        pass
    
    return build_pipeline(base_steps)
```

**Approach 4: Runtime Composition**
```python
def build_from_config(config: dict):
    """Build graph from runtime configuration"""
    
    workflow = StateGraph(State)
    
    # Add nodes from config
    for node_config in config["nodes"]:
        node_fn = get_function_by_name(node_config["function"])
        workflow.add_node(node_config["name"], node_fn)
    
    # Add edges from config
    for edge in config["edges"]:
        workflow.add_edge(edge["from"], edge["to"])
    
    return workflow.compile()
```

#### Dynamic Routing vs Dynamic Structure

**Dynamic Routing (what we've done):**
```python
# Graph structure is fixed
# Path through graph is dynamic
workflow.add_conditional_edges("router", route_fn, {...})
→ Structure: A → [B or C] → D (fixed)
→ Path: Determined at runtime
```

**Dynamic Structure (new):**
```python
# Graph structure itself is dynamic
# Built at runtime based on input
graph = build_graph_for(user_request)
→ Structure: Created on-the-fly
→ Different structure for different requests
```

#### When to Use Dynamic Graphs

✅ **Use Dynamic Graphs When:**
- Workflow varies significantly by input
- Users configure their own workflows
- Plugin/extension architecture
- A/B testing different workflows
- Workflow evolves based on results

❌ **Don't Use When:**
- Static workflow is sufficient
- Adds unnecessary complexity
- Debugging becomes too hard
- Performance is critical (compilation overhead)

#### Challenges with Dynamic Graphs

**1. Validation:**
```
How do you validate a graph you haven't built yet?
→ Need schema validation, cycle detection at runtime
```

**2. Testing:**
```
Can't test all possible dynamic configurations
→ Test graph builder logic + templates
```

**3. Debugging:**
```
Different graph each time makes debugging harder
→ Log graph structure, visualize what was built
```

**4. Performance:**
```
Building and compiling graphs has overhead
→ Cache compiled graphs when possible
```

---

### Implementation: Dynamic Graph Construction

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Callable, Any
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DYNAMIC GRAPH CONSTRUCTION ====================

# Common state for all dynamic graphs
class DynamicState(TypedDict):
    """State for dynamic workflows"""
    messages: Annotated[Sequence[BaseMessage], add]
    data: str
    processing_log: Annotated[List[str], add]
    result: str

# ==================== NODE LIBRARY ====================

# Library of available node functions
def ingest_node(state: DynamicState) -> dict:
    """Ingest data"""
    logger.info("  [NODE] Ingest")
    return {
        "processing_log": ["Ingested data"],
        "messages": [AIMessage(content="Data ingested")]
    }

def validate_node(state: DynamicState) -> dict:
    """Validate data"""
    logger.info("  [NODE] Validate")
    return {
        "processing_log": ["Validated data"],
        "messages": [AIMessage(content="Data validated")]
    }

def clean_node(state: DynamicState) -> dict:
    """Clean data"""
    logger.info("  [NODE] Clean")
    cleaned = state["data"].strip().lower()
    return {
        "data": cleaned,
        "processing_log": ["Cleaned data"],
        "messages": [AIMessage(content="Data cleaned")]
    }

def transform_node(state: DynamicState) -> dict:
    """Transform data"""
    logger.info("  [NODE] Transform")
    transformed = state["data"].upper()
    return {
        "data": transformed,
        "processing_log": ["Transformed data"],
        "messages": [AIMessage(content="Data transformed")]
    }

def enrich_node(state: DynamicState) -> dict:
    """Enrich data"""
    logger.info("  [NODE] Enrich")
    enriched = f"[ENRICHED] {state['data']}"
    return {
        "data": enriched,
        "processing_log": ["Enriched data"],
        "messages": [AIMessage(content="Data enriched")]
    }

def analyze_node(state: DynamicState) -> dict:
    """Analyze data"""
    logger.info("  [NODE] Analyze")
    analysis = f"Analysis: Length={len(state['data'])}, Words={len(state['data'].split())}"
    return {
        "processing_log": [f"Analyzed: {analysis}"],
        "messages": [AIMessage(content=f"Analysis complete: {analysis}")]
    }

def load_node(state: DynamicState) -> dict:
    """Load/save data"""
    logger.info("  [NODE] Load")
    return {
        "processing_log": ["Loaded/saved data"],
        "result": state["data"],
        "messages": [AIMessage(content="Data loaded")]
    }

def summarize_node(state: DynamicState) -> dict:
    """Summarize processing"""
    logger.info("  [NODE] Summarize")
    
    summary = f"""Processing Complete:
Steps: {len(state['processing_log'])}
Log: {' → '.join(state['processing_log'])}
Final Data: {state['data'][:50]}..."""
    
    return {
        "result": summary,
        "messages": [AIMessage(content=summary)]
    }

# Node registry
NODE_REGISTRY = {
    "ingest": ingest_node,
    "validate": validate_node,
    "clean": clean_node,
    "transform": transform_node,
    "enrich": enrich_node,
    "analyze": analyze_node,
    "load": load_node,
    "summarize": summarize_node
}

# ==================== DYNAMIC GRAPH BUILDERS ====================

def build_sequential_pipeline(steps: List[str]) -> StateGraph:
    """Build a sequential pipeline from list of steps"""
    
    logger.info(f"🔨 Building sequential pipeline with {len(steps)} steps")
    
    workflow = StateGraph(DynamicState)
    
    # Add all nodes
    for step in steps:
        if step not in NODE_REGISTRY:
            raise ValueError(f"Unknown step: {step}")
        workflow.add_node(step, NODE_REGISTRY[step])
    
    # Connect sequentially
    workflow.set_entry_point(steps[0])
    for i in range(len(steps) - 1):
        workflow.add_edge(steps[i], steps[i + 1])
    workflow.add_edge(steps[-1], END)
    
    logger.info(f"✅ Pipeline built: {' → '.join(steps)}")
    
    return workflow

def build_from_template(template_name: str, options: Dict[str, bool] = None) -> StateGraph:
    """Build graph from pre-defined template"""
    
    if options is None:
        options = {}
    
    logger.info(f"🔨 Building from template: {template_name}")
    
    # Pre-defined templates
    templates = {
        "basic_etl": ["ingest", "clean", "transform", "load"],
        "data_pipeline": ["ingest", "validate", "clean", "transform", "load"],
        "analysis_pipeline": ["ingest", "clean", "analyze", "summarize"],
        "full_pipeline": ["ingest", "validate", "clean", "transform", "enrich", "analyze", "load", "summarize"]
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")
    
    steps = templates[template_name].copy()
    
    # Apply options
    if options.get("skip_validation") and "validate" in steps:
        steps.remove("validate")
        logger.info("  Option: Skipping validation")
    
    if options.get("skip_enrichment") and "enrich" in steps:
        steps.remove("enrich")
        logger.info("  Option: Skipping enrichment")
    
    if options.get("include_analysis") and "analyze" not in steps:
        # Insert before last step
        steps.insert(-1, "analyze")
        logger.info("  Option: Including analysis")
    
    return build_sequential_pipeline(steps)

def build_from_config(config: dict) -> StateGraph:
    """Build graph from JSON-like configuration"""
    
    logger.info("🔨 Building from configuration")
    
    workflow = StateGraph(DynamicState)
    
    # Add nodes
    for node in config["nodes"]:
        node_name = node["name"]
        node_function = node["function"]
        
        if node_function not in NODE_REGISTRY:
            raise ValueError(f"Unknown function: {node_function}")
        
        workflow.add_node(node_name, NODE_REGISTRY[node_function])
        logger.info(f"  Added node: {node_name} (function: {node_function})")
    
    # Set entry point
    entry = config.get("entry_point", config["nodes"][0]["name"])
    workflow.set_entry_point(entry)
    logger.info(f"  Entry point: {entry}")
    
    # Add edges
    for edge in config["edges"]:
        workflow.add_edge(edge["from"], edge["to"])
        logger.info(f"  Added edge: {edge['from']} → {edge['to']}")
    
    # Handle terminal nodes
    terminal_nodes = set(node["name"] for node in config["nodes"])
    for edge in config["edges"]:
        terminal_nodes.discard(edge["from"])
    
    for terminal in terminal_nodes:
        workflow.add_edge(terminal, END)
        logger.info(f"  Terminal node: {terminal}")
    
    return workflow

def build_adaptive_pipeline(data_characteristics: dict) -> StateGraph:
    """Build pipeline that adapts to data characteristics"""
    
    logger.info("🔨 Building adaptive pipeline")
    
    steps = ["ingest"]
    
    # Adapt based on data characteristics
    if data_characteristics.get("needs_validation", False):
        steps.append("validate")
        logger.info("  Adaptive: Including validation (data needs validation)")
    
    if data_characteristics.get("is_dirty", False):
        steps.append("clean")
        logger.info("  Adaptive: Including cleaning (data is dirty)")
    
    # Always transform
    steps.append("transform")
    
    if data_characteristics.get("complexity", "low") == "high":
        steps.append("enrich")
        logger.info("  Adaptive: Including enrichment (high complexity)")
    
    if data_characteristics.get("requires_analysis", False):
        steps.append("analyze")
        logger.info("  Adaptive: Including analysis (analysis required)")
    
    # Always load and summarize
    steps.extend(["load", "summarize"])
    
    return build_sequential_pipeline(steps)

# ==================== GRAPH EXECUTOR ====================

def execute_dynamic_graph(
    graph_builder: Callable,
    builder_args: Any,
    input_data: str
) -> dict:
    """Build and execute dynamic graph"""
    
    logger.info("="*60)
    logger.info("DYNAMIC GRAPH EXECUTION")
    logger.info("="*60)
    
    # Build graph
    if isinstance(builder_args, dict):
        workflow = graph_builder(**builder_args)
    elif isinstance(builder_args, list):
        workflow = graph_builder(*builder_args)
    else:
        workflow = graph_builder(builder_args)
    
    # Compile
    compiled = workflow.compile()
    logger.info("✅ Graph compiled")
    
    # Execute
    logger.info("▶️  Executing graph")
    logger.info("-"*60)
    
    result = compiled.invoke({
        "messages": [HumanMessage(content="Start processing")],
        "data": input_data,
        "processing_log": [],
        "result": ""
    })
    
    logger.info("-"*60)
    logger.info("✅ Execution complete")
    
    return result

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DYNAMIC GRAPH CONSTRUCTION EXAMPLES")
    print("="*60)
    
    test_data = "  Sample Data for Processing  "
    
    # Example 1: Sequential pipeline
    print("\n" + "="*60)
    print("EXAMPLE 1: Sequential Pipeline")
    print("="*60)
    
    result1 = execute_dynamic_graph(
        graph_builder=build_sequential_pipeline,
        builder_args=["ingest", "clean", "transform", "summarize"],
        input_data=test_data
    )
    print(f"\nResult:\n{result1['result']}")
    
    # Example 2: Template-based
    print("\n" + "="*60)
    print("EXAMPLE 2: Template-Based (basic_etl)")
    print("="*60)
    
    result2 = execute_dynamic_graph(
        graph_builder=build_from_template,
        builder_args={"template_name": "basic_etl", "options": {}},
        input_data=test_data
    )
    print(f"\nResult:\n{result2['result']}")
    
    # Example 3: Template with options
    print("\n" + "="*60)
    print("EXAMPLE 3: Template with Options")
    print("="*60)
    
    result3 = execute_dynamic_graph(
        graph_builder=build_from_template,
        builder_args={
            "template_name": "data_pipeline",
            "options": {"skip_validation": True, "include_analysis": True}
        },
        input_data=test_data
    )
    print(f"\nResult:\n{result3['result']}")
    
    # Example 4: Configuration-based
    print("\n" + "="*60)
    print("EXAMPLE 4: Configuration-Based")
    print("="*60)
    
    config = {
        "nodes": [
            {"name": "start", "function": "ingest"},
            {"name": "cleaner", "function": "clean"},
            {"name": "analyzer", "function": "analyze"},
            {"name": "finish", "function": "summarize"}
        ],
        "edges": [
            {"from": "start", "to": "cleaner"},
            {"from": "cleaner", "to": "analyzer"},
            {"from": "analyzer", "to": "finish"}
        ],
        "entry_point": "start"
    }
    
    result4 = execute_dynamic_graph(
        graph_builder=build_from_config,
        builder_args=config,
        input_data=test_data
    )
    print(f"\nResult:\n{result4['result']}")
    
    # Example 5: Adaptive pipeline
    print("\n" + "="*60)
    print("EXAMPLE 5: Adaptive Pipeline")
    print("="*60)
    
    characteristics = {
        "needs_validation": True,
        "is_dirty": True,
        "complexity": "high",
        "requires_analysis": True
    }
    
    result5 = execute_dynamic_graph(
        graph_builder=build_adaptive_pipeline,
        builder_args=characteristics,
        input_data=test_data
    )
    print(f"\nResult:\n{result5['result']}")
```

---

## 🌊 Part 5: Streaming and Async Patterns

### Theory: Streaming in LangGraph

#### What Is Streaming?

**Streaming** means producing output **incrementally** as the workflow executes, rather than waiting for everything to complete.

```
Non-Streaming:
┌─────┐   ┌─────┐   ┌─────┐
│  A  │──→│  B  │──→│  C  │
└─────┘   └─────┘   └─────┘
          ↓
      [Wait for all]
          ↓
      Full output

Streaming:
┌─────┐   ┌─────┐   ┌─────┐
│  A  │──→│  B  │──→│  C  │
└──┬──┘   └──┬──┘   └──┬──┘
   ↓         ↓         ↓
Output 1  Output 2  Output 3
(Real-time output as nodes complete)
```

#### Why Streaming?

**1. Better User Experience:**
```
User sees progress immediately
Not staring at blank screen
Feels faster even if total time is same
```

**2. Early Access to Results:**
```
Start using partial results before completion
Example: Display first search results while still searching
```

**3. Long-Running Workflows:**
```
Workflows that take minutes/hours
User needs to see it's working
Can cancel if going wrong direction
```

**4. Real-Time Applications:**
```
Chat applications (streaming responses)
Live data processing
Progressive rendering
```

#### LangGraph Streaming Modes

**Mode 1: Stream Values**
```python
for chunk in graph.stream(input_data):
    print(chunk)
    # Each chunk is the full state after each node

Output:
{'node': 'node_a', 'state': {...}}
{'node': 'node_b', 'state': {...}}
{'node': 'node_c', 'state': {...}}
```

**Mode 2: Stream Updates**
```python
for chunk in graph.stream(input_data, stream_mode="updates"):
    print(chunk)
    # Each chunk is only the UPDATE from that node

Output:
{'node_a': {'field1': 'new_value'}}
{'node_b': {'field2': 'another_value'}}
```

**Mode 3: Stream Messages**
```python
for chunk in graph.stream(input_data, stream_mode="messages"):
    print(chunk)
    # Each chunk is individual messages

Output:
[AIMessage(content="Step 1 complete")]
[AIMessage(content="Step 2 complete")]
```

#### Async Patterns

**Asynchronous execution** allows the program to do other work while waiting for I/O operations:

```python
# Synchronous (blocking)
result1 = call_api_1()  # Wait...
result2 = call_api_2()  # Wait...
result3 = call_api_3()  # Wait...
# Total time: time1 + time2 + time3

# Asynchronous (non-blocking)
results = await asyncio.gather(
    call_api_1(),
    call_api_2(),
    call_api_3()
)
# Total time: max(time1, time2, time3)
```

#### When to Use Streaming

✅ **Use Streaming When:**
- Long-running workflows (>5 seconds)
- User-facing applications (chat, UI)
- Want to show progress
- Need early access to partial results
- Real-time data processing

❌ **Don't Use When:**
- Simple, fast workflows (<1 second)
- Batch processing where order matters
- Streaming adds unnecessary complexity
- Need atomic all-or-nothing results

#### When to Use Async

✅ **Use Async When:**
- Making multiple I/O calls (APIs, databases)
- Want concurrent execution
- High-throughput requirements
- Need to handle many requests simultaneously

❌ **Don't Use When:**
- CPU-bound tasks (computation heavy)
- Simple sequential workflows
- Adds complexity without benefit
- Team lacks async expertise

#### Streaming Best Practices

**1. Provide Meaningful Updates:**
```python
# Bad: Just "processing"
yield {"status": "processing"}

# Good: Specific progress
yield {"status": "processing", "step": "analyzing", "progress": 0.3}
```

**2. Handle Errors in Stream:**
```python
try:
    for chunk in graph.stream(input_data):
        yield chunk
except Exception as e:
    yield {"error": str(e), "status": "failed"}
```

**3. Indicate Completion:**
```python
for chunk in graph.stream(input_data):
    yield chunk
yield {"status": "complete", "final_result": result}
```

**4. Allow Cancellation:**
```python
for chunk in graph.stream(input_data):
    if user_cancelled:
        break
    yield chunk
```

---

### Implementation: Streaming and Async System

```python
from typing import TypedDict, Annotated, Sequence, List, AsyncIterator
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== STREAMING EXAMPLES ====================

class StreamingState(TypedDict):
    """State for streaming examples"""
    messages: Annotated[Sequence[BaseMessage], add]
    step_count: Annotated[int, add]
    progress: float

# Slow nodes to demonstrate streaming
def slow_step_1(state: StreamingState) -> dict:
    """First slow step"""
    logger.info("Step 1: Processing...")
    time.sleep(1.5)
    
    return {
        "messages": [AIMessage(content="Step 1 complete")],
        "step_count": 1,
        "progress": 0.33
    }

def slow_step_2(state: StreamingState) -> dict:
    """Second slow step"""
    logger.info("Step 2: Processing...")
    time.sleep(1.5)
    
    return {
        "messages": [AIMessage(content="Step 2 complete")],
        "step_count": 1,
        "progress": 0.66
    }

def slow_step_3(state: StreamingState) -> dict:
    """Third slow step"""
    logger.info("Step 3: Processing...")
    time.sleep(1.5)
    
    return {
        "messages": [AIMessage(content="Step 3 complete")],
        "step_count": 1,
        "progress": 1.0
    }

# Build streaming workflow
streaming_workflow = StateGraph(StreamingState)
streaming_workflow.add_node("step1", slow_step_1)
streaming_workflow.add_node("step2", slow_step_2)
streaming_workflow.add_node("step3", slow_step_3)
streaming_workflow.set_entry_point("step1")
streaming_workflow.add_edge("step1", "step2")
streaming_workflow.add_edge("step2", "step3")
streaming_workflow.add_edge("step3", END)

streaming_system = streaming_workflow.compile()

# ==================== STREAMING DEMONSTRATIONS ====================

def demo_non_streaming():
    """Demonstrate non-streaming execution"""
    print("\n" + "="*60)
    print("NON-STREAMING (Traditional)")
    print("="*60)
    print("Waiting for all steps to complete...")
    
    start_time = time.time()
    
    result = streaming_system.invoke({
        "messages": [HumanMessage(content="Start")],
        "step_count": 0,
        "progress": 0.0
    })
    
    end_time = time.time()
    
    print(f"\n✅ All steps complete!")
    print(f"Total time: {end_time - start_time:.1f}s")
    print(f"Steps completed: {result['step_count']}")
    print(f"Progress: {result['progress']*100:.0f}%")

def demo_streaming_values():
    """Demonstrate streaming with values mode"""
    print("\n" + "="*60)
    print("STREAMING - Values Mode")
    print("="*60)
    print("Streaming state after each node...\n")
    
    start_time = time.time()
    
    for i, chunk in enumerate(streaming_system.stream({
        "messages": [HumanMessage(content="Start")],
        "step_count": 0,
        "progress": 0.0
    })):
        current_time = time.time() - start_time
        
        # Each chunk contains node name and full state
        node_name = list(chunk.keys())[0]
        node_state = chunk[node_name]
        
        print(f"[{current_time:.1f}s] Node '{node_name}' completed:")
        print(f"  Progress: {node_state.get('progress', 0)*100:.0f}%")
        print(f"  Steps so far: {node_state.get('step_count', 0)}")
        
        if node_state.get('messages'):
            last_msg = node_state['messages'][-1]
            print(f"  Message: {last_msg.content}")
        print()
    
    print(f"✅ Streaming complete! Total time: {time.time() - start_time:.1f}s")

def demo_streaming_updates():
    """Demonstrate streaming with updates mode"""
    print("\n" + "="*60)
    print("STREAMING - Updates Mode")
    print("="*60)
    print("Streaming only updates from each node...\n")
    
    start_time = time.time()
    
    for i, chunk in enumerate(streaming_system.stream(
        {
            "messages": [HumanMessage(content="Start")],
            "step_count": 0,
            "progress": 0.0
        },
        stream_mode="updates"
    )):
        current_time = time.time() - start_time
        
        # Each chunk contains only the updates from that node
        node_name = list(chunk.keys())[0]
        updates = chunk[node_name]
        
        print(f"[{current_time:.1f}s] Update from '{node_name}':")
        for key, value in updates.items():
            if key == 'messages':
                print(f"  {key}: {value[-1].content}")
            else:
                print(f"  {key}: {value}")
        print()
    
    print(f"✅ Streaming complete! Total time: {time.time() - start_time:.1f}s")

# ==================== ASYNC PATTERNS ====================

class AsyncState(TypedDict):
    """State for async examples"""
    messages: Annotated[Sequence[BaseMessage], add]
    results: Annotated[List[str], add]

# Async node functions
async def async_api_call_1(state: AsyncState) -> dict:
    """Simulated async API call"""
    logger.info("API Call 1: Starting...")
    await asyncio.sleep(2.0)  # Simulate network delay
    logger.info("API Call 1: Complete")
    
    return {
        "results": ["Result from API 1"],
        "messages": [AIMessage(content="API 1 complete")]
    }

async def async_api_call_2(state: AsyncState) -> dict:
    """Simulated async API call"""
    logger.info("API Call 2: Starting...")
    await asyncio.sleep(1.5)  # Simulate network delay
    logger.info("API Call 2: Complete")
    
    return {
        "results": ["Result from API 2"],
        "messages": [AIMessage(content="API 2 complete")]
    }

async def async_api_call_3(state: AsyncState) -> dict:
    """Simulated async API call"""
    logger.info("API Call 3: Starting...")
    await asyncio.sleep(1.8)  # Simulate network delay
    logger.info("API Call 3: Complete")
    
    return {
        "results": ["Result from API 3"],
        "messages": [AIMessage(content="API 3 complete")]
    }

# Note: LangGraph's current version may not fully support async nodes
# This demonstrates the pattern conceptually

async def demo_async_benefits():
    """Demonstrate benefits of async execution"""
    print("\n" + "="*60)
    print("ASYNC vs SYNC COMPARISON")
    print("="*60)
    
    # Synchronous simulation
    print("\n--- SYNCHRONOUS (one at a time) ---")
    sync_start = time.time()
    
    print("Calling API 1...")
    await asyncio.sleep(2.0)
    print("Calling API 2...")
    await asyncio.sleep(1.5)
    print("Calling API 3...")
    await asyncio.sleep(1.8)
    
    sync_time = time.time() - sync_start
    print(f"Synchronous total time: {sync_time:.1f}s")
    
    # Asynchronous
    print("\n--- ASYNCHRONOUS (concurrent) ---")
    async_start = time.time()
    
    print("Starting all API calls concurrently...")
    await asyncio.gather(
        asyncio.sleep(2.0),  # API 1
        asyncio.sleep(1.5),  # API 2
        asyncio.sleep(1.8)   # API 3
    )
    
    async_time = time.time() - async_start
    print(f"Asynchronous total time: {async_time:.1f}s")
    
    print(f"\n📊 Speed improvement: {sync_time/async_time:.1f}x faster")
    print(f"Time saved: {sync_time - async_time:.1f}s")

# ==================== PROGRESSIVE STREAMING ====================

def progressive_streaming_example():
    """Example of progressive output in streaming"""
    print("\n" + "="*60)
    print("PROGRESSIVE STREAMING")
    print("="*60)
    print("Showing incremental progress...\n")
    
    total_steps = 5
    
    for i in range(1, total_steps + 1):
        progress = i / total_steps
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Stream progressive update
        print(f"[{progress*100:5.1f}%] Step {i}/{total_steps} complete", end='')
        print(" " + "█" * int(progress * 30), flush=True)
    
    print("\n✅ Processing complete!")

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STREAMING AND ASYNC PATTERNS")
    print("="*60)
    
    # Demo 1: Non-streaming
    demo_non_streaming()
    
    # Demo 2: Streaming values
    demo_streaming_values()
    
    # Demo 3: Streaming updates
    demo_streaming_updates()
    
    # Demo 4: Progressive streaming
    progressive_streaming_example()
    
    # Demo 5: Async benefits
    print("\nRunning async demonstration...")
    asyncio.run(demo_async_benefits())
```

---

## 📋 Best Practices Summary

### Custom Reducers

**✅ DO:**
- Use `operator.add` for accumulation (lists, numbers)
- Create custom reducers for complex merge logic
- Document reducer behavior clearly
- Test reducer edge cases

**❌ DON'T:**
- Over-complicate with unnecessary reducers
- Forget that no annotation = replace
- Mutate existing state in reducers
- Create reducers with side effects

---

### Branching and Parallel

**✅ DO:**
- Use branching for conditional logic
- Use parallel for independent tasks
- Ensure parallel paths can merge safely
- Test all branch paths

**❌ DON'T:**
- Branch when simple if-statement would work
- Parallelize tasks with dependencies
- Forget to handle all branch outcomes
- Create race conditions in parallel execution

---

### Sub-Graphs

**✅ DO:**
- Use for logical modularity
- Test sub-graphs independently
- Document state mapping between parent/child
- Keep sub-graphs focused and cohesive

**❌ DON'T:**
- Over-nest (2-3 levels max)
- Create circular dependencies
- Forget to map state properly
- Use when simple function would work

---

### Dynamic Graphs

**✅ DO:**
- Validate graph structure at runtime
- Cache compiled graphs when possible
- Log the graph structure built
- Provide templates for common patterns

**❌ DON'T:**
- Build unnecessarily complex graphs
- Skip validation
- Forget about performance overhead
- Make debugging impossible

---

### Streaming and Async

**✅ DO:**
- Stream for long-running workflows
- Provide meaningful progress updates
- Use async for I/O-bound tasks
- Handle errors in streams gracefully

**❌ DON'T:**
- Stream when not needed
- Use async for CPU-bound work
- Forget to indicate completion
- Block the event loop

---

## ✅ Chapter 16 Complete!

**You now understand:**
- ✅ Custom reducers for state merging (add, max, custom functions)
- ✅ Branching with conditional edges
- ✅ Parallel execution patterns (fan-out/fan-in)
- ✅ Sub-graphs for modularity and composition
- ✅ Dynamic graph construction at runtime
- ✅ Streaming for progressive output
- ✅ Async patterns for concurrent execution
- ✅ When to use each advanced pattern
- ✅ Best practices for advanced state management

**Key Takeaways:**
- Custom reducers control how state accumulates
- Branching enables conditional workflows
- Parallel execution speeds up independent tasks
- Sub-graphs enable modular, reusable components
- Dynamic graphs adapt to runtime conditions
- Streaming provides real-time feedback
- Async improves throughput for I/O-bound work

---

**Ready for Chapter 17?**

**Chapter 17: Human-in-the-Loop Systems** will cover:
- Interrupt patterns and breakpoints
- Approval workflows
- Dynamic user input
- Editing and resuming execution
- Audit trails and compliance

Just say "Continue to Chapter 17" when ready!