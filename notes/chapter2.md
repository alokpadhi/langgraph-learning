# Chapter 2: LangGraph Core Concepts & Architecture

## 🎯 The Problem: Building Robust, Maintainable Agents

You've built your first LangGraph agent in Chapter 1. But now you need to:
- Handle complex state with multiple data types
- Manage conversations with proper message history
- Make your agent resumable (pause and continue later)
- Handle errors gracefully
- Debug when things go wrong

Let's learn how to do this properly with LangGraph 1.0.x.

---

## 📊 StateGraph vs MessageGraph: Choosing Your Foundation

LangGraph provides **two graph types** for different use cases:

### 1. **StateGraph** - For Custom State Management

**Use when:** You need complete control over state structure

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph

class CustomState(TypedDict):
    messages: list
    user_id: str
    iteration_count: int
    context: dict

graph = StateGraph(CustomState)
```

### 2. **MessageGraph** - For Simple Chat Applications

**Use when:** You only need message history (simpler, more opinionated)

```python
from langgraph.graph import MessageGraph

# State is just a list of messages
graph = MessageGraph()
```
In Langrgraph 1.0.0 MessageGraph is depricated. MessageGraph is a subclass of StateGraph whose entire state is a single, append-only* list of messages. Each node in a MessageGraph takes a list of messages as input and returns zero or more messages as output. The add_messages function is used to merge the output messages from each node into the existing list of messages in the graph's state.

**Decision Framework:**
```
Need only chat history? → MessageGraph
Need custom fields/logic? → StateGraph
Building complex agents?  → StateGraph (more flexible)
Quick chatbot prototype?  → MessageGraph (faster)
```

---

## 🏗️ Deep Dive: StateGraph Architecture

### The State Schema - Your Agent's Memory

State is the **single source of truth** that flows through your graph.

#### Level 1: Basic State (Minimal Example)

```python
from typing import TypedDict

class SimpleState(TypedDict):
    input: str
    output: str
```

**Problem:** What if a node only updates `output` but not `input`? We need to handle partial updates.

#### Level 2: State with Reducers (The Right Way)

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    # Reducer: add (append to list)
    messages: Annotated[list, add]
    
    # Reducer: default (replace value)
    user_id: str
    
    # Reducer: add (sum integers)
    iteration: Annotated[int, add]
```

**What are reducers?**
- Functions that determine **how to merge** updates with existing state
- `add` for lists → appends
- `add` for ints → sums
- No annotation → replaces value

**Visual explanation:**
```
Current state: {"messages": [msg1, msg2], "iteration": 1}
Node returns: {"messages": [msg3], "iteration": 1}

With 'add' reducer:
Result: {"messages": [msg1, msg2, msg3], "iteration": 2}

Without reducer (default replace):
Result: {"messages": [msg3], "iteration": 1}
```

---

### Complete Example: StateGraph with Proper State Management

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Step 1: Define state with appropriate reducers
class AgentState(TypedDict):
    # Messages list - append new messages
    messages: Annotated[list, add]
    
    # Current step in workflow
    current_step: str
    
    # Iteration counter - increment
    iteration: Annotated[int, add]
    
    # Max iterations allowed
    max_iterations: int
    
    # Error tracking
    error: str | None

# Step 2: Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Step 3: Define nodes
def process_input(state: AgentState) -> dict:
    """Process user input and generate initial response"""
    try:
        response = llm.invoke(state["messages"])
        
        return {
            "messages": [response],  # Will be appended via 'add' reducer
            "current_step": "processed",
            "iteration": 1  # Will be added to existing value
        }
    except Exception as e:
        return {
            "error": str(e),
            "current_step": "error"
        }

def review_output(state: AgentState) -> dict:
    """Review the output for quality"""
    last_message = state["messages"][-1].content
    
    review_prompt = f"""Review this response for accuracy and completeness:
    
    Response: {last_message}
    
    Rate from 1-10 and provide brief feedback."""
    
    review = llm.invoke([SystemMessage(content=review_prompt)])
    
    return {
        "messages": [SystemMessage(content=f"Review: {review.content}")],
        "current_step": "reviewed"
    }

def improve_output(state: AgentState) -> dict:
    """Improve based on review"""
    # Get original question and review
    original_q = state["messages"][0].content
    review = state["messages"][-1].content
    
    improve_prompt = f"""Original question: {original_q}
    
    Review feedback: {review}
    
    Provide an improved response addressing the feedback."""
    
    improved = llm.invoke([SystemMessage(content=improve_prompt)])
    
    return {
        "messages": [improved],
        "iteration": 1  # Increment counter
    }

# Step 4: Define routing logic
def should_continue(state: AgentState) -> Literal["improve", "review", "end"]:
    """Decide next step based on state"""
    
    # Check for errors
    if state.get("error"):
        return "end"
    
    # Check iteration limit
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    
    # Route based on current step
    if state["current_step"] == "processed":
        return "review"
    elif state["current_step"] == "reviewed":
        # Simple logic: improve if iteration < max
        if state["iteration"] < state["max_iterations"]:
            return "improve"
        return "end"
    
    return "end"

# Step 5: Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("process", process_input)
workflow.add_node("review", review_output)
workflow.add_node("improve", improve_output)

# Set entry point
workflow.set_entry_point("process")

# Add conditional edges
workflow.add_conditional_edges(
    "process",
    should_continue,
    {
        "review": "review",
        "end": END
    }
)

workflow.add_conditional_edges(
    "review",
    should_continue,
    {
        "improve": "improve",
        "end": END
    }
)

workflow.add_edge("improve", "review")  # Loop back to review

# Step 6: Compile
app = workflow.compile()

# Step 7: Use it
result = app.invoke({
    "messages": [HumanMessage(content="Explain neural networks in simple terms")],
    "current_step": "init",
    "iteration": 0,
    "max_iterations": 2,
    "error": None
})

# Print results
print(f"Total iterations: {result['iteration']}")
print(f"\nFinal response:")
print(result["messages"][-1].content)
```

**Graph visualization:**
```
        process
           ↓
    should_continue?
       /      \
   review    END
      ↓
should_continue?
    /     \
improve   END
   ↓
review (loop)
```

---

## 🎯 Nodes: The Building Blocks

### Node Anatomy

```python
def my_node(state: StateType) -> dict:
    """
    A node is a function that:
    1. Takes the full state as input
    2. Performs some work (LLM call, tool use, computation)
    3. Returns a PARTIAL update (will be merged with state)
    """
    # Read from state
    data = state["some_field"]
    
    # Do work
    result = process(data)
    
    # Return partial update
    return {
        "some_field": result,
        "updated_at": time.time()
    }
```

### Node Best Practices

#### ✅ Good Node Design

```python
def good_node(state: AgentState) -> dict:
    """Clear, focused responsibility"""
    try:
        # 1. Extract what you need
        user_input = state["messages"][-1].content
        
        # 2. Do one thing well
        result = llm.invoke(user_input)
        
        # 3. Return minimal update
        return {"messages": [result]}
        
    except Exception as e:
        # 4. Handle errors gracefully
        return {"error": str(e)}
```

#### ❌ Bad Node Design

```python
def bad_node(state: AgentState) -> dict:
    """Multiple responsibilities, unclear purpose"""
    # ❌ Doing too much in one node
    result1 = llm.invoke(state["messages"])
    result2 = search_tool.run(result1)
    result3 = llm.invoke(result2)
    
    # ❌ Mutating state directly
    state["messages"].append(result3)
    
    # ❌ Returning full state instead of update
    return state
```

**Split into multiple nodes instead:**
```python
def generate_query(state): ...
def search_web(state): ...
def synthesize_results(state): ...
```

---

## 🔀 Edges: Controlling Flow

### 1. **Fixed Edges** - Unconditional Transitions

```python
# Always go from node_a to node_b
workflow.add_edge("node_a", "node_b")

# End after this node
workflow.add_edge("final_node", END)
```

### 2. **Conditional Edges** - Dynamic Routing

```python
def router(state: AgentState) -> str:
    """
    Router function that returns the name of the next node
    """
    if state["error"]:
        return "error_handler"
    elif state["needs_search"]:
        return "search"
    else:
        return "respond"

workflow.add_conditional_edges(
    "decision_node",  # From this node
    router,           # Use this function to decide
    {
        "error_handler": "handle_error",
        "search": "search_node",
        "respond": "respond_node"
    }
)
```

### 3. **Entry Point** - Where Execution Starts

```python
# Set the starting node
workflow.set_entry_point("initial_node")
```

### Complete Routing Example

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END

class State(TypedDict):
    query: str
    query_type: Literal["factual", "creative", "computational"] | None
    result: str

def classify_query(state: State) -> dict:
    """Classify the query type"""
    query = state["query"].lower()
    
    if any(word in query for word in ["calculate", "compute", "math"]):
        qtype = "computational"
    elif any(word in query for word in ["story", "poem", "creative"]):
        qtype = "creative"
    else:
        qtype = "factual"
    
    return {"query_type": qtype}

def route_query(state: State) -> str:
    """Route based on query type"""
    return state["query_type"]  # Returns "factual", "creative", or "computational"

def handle_factual(state: State) -> dict:
    return {"result": f"Factual answer for: {state['query']}"}

def handle_creative(state: State) -> dict:
    return {"result": f"Creative response for: {state['query']}"}

def handle_computational(state: State) -> dict:
    return {"result": f"Computation for: {state['query']}"}

# Build graph
workflow = StateGraph(State)

workflow.add_node("classify", classify_query)
workflow.add_node("factual", handle_factual)
workflow.add_node("creative", handle_creative)
workflow.add_node("computational", handle_computational)

workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    route_query,
    {
        "factual": "factual",
        "creative": "creative",
        "computational": "computational"
    }
)

# All paths lead to END
workflow.add_edge("factual", END)
workflow.add_edge("creative", END)
workflow.add_edge("computational", END)

app = workflow.compile()

# Test different query types
test_queries = [
    "What is the capital of France?",
    "Write me a short poem about clouds",
    "Calculate 123 * 456"
]

for query in test_queries:
    result = app.invoke({"query": query, "query_type": None, "result": ""})
    print(f"\nQuery: {query}")
    print(f"Type: {result['query_type']}")
    print(f"Result: {result['result']}")
```

---

## 🔄 The Compilation and Execution Model

### What Happens When You Compile?

```python
app = workflow.compile()
```

**Behind the scenes:**
1. Validates graph structure (no orphaned nodes, valid edges)
2. Creates execution engine
3. Sets up state management
4. Prepares checkpointing (if enabled)
5. Returns a runnable application

### Execution Modes

#### Mode 1: **invoke()** - Synchronous, Returns Final State

```python
result = app.invoke(initial_state)
# Returns: Final state after all nodes execute
```

#### Mode 2: **stream()** - Iterate Over Each Step

```python
for step in app.stream(initial_state):
    print(f"Step: {step}")
    # Each iteration returns state after one node
```

#### Mode 3: **astream()** - Async Streaming

```python
async for step in app.astream(initial_state):
    print(f"Async step: {step}")
```

### Practical Streaming Example

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict):
    count: int
    step_name: str

def step1(state: State) -> dict:
    return {"count": state["count"] + 1, "step_name": "step1"}

def step2(state: State) -> dict:
    return {"count": state["count"] * 2, "step_name": "step2"}

def step3(state: State) -> dict:
    return {"count": state["count"] - 1, "step_name": "step3"}

workflow = StateGraph(State)
workflow.add_node("s1", step1)
workflow.add_node("s2", step2)
workflow.add_node("s3", step3)

workflow.set_entry_point("s1")
workflow.add_edge("s1", "s2")
workflow.add_edge("s2", "s3")
workflow.add_edge("s3", END)

app = workflow.compile()

print("=== Using invoke() ===")
result = app.invoke({"count": 5, "step_name": "init"})
print(f"Final result: {result}")

print("\n=== Using stream() ===")
for i, step in enumerate(app.stream({"count": 5, "step_name": "init"})):
    print(f"After step {i+1}: {step}")
```

**Output:**
```
=== Using invoke() ===
Final result: {'count': 11, 'step_name': 'step3'}

=== Using stream() ===
After step 1: {'count': 6, 'step_name': 'step1'}
After step 2: {'count': 12, 'step_name': 'step2'}
After step 3: {'count': 11, 'step_name': 'step3'}
```

---

## 💾 Checkpointing and Persistence

### Why Checkpointing?

**Problems without checkpointing:**
- Can't pause and resume long-running agents
- Can't recover from crashes
- Can't inspect intermediate states for debugging
- Can't implement human-in-the-loop patterns

### Built-in Checkpointer: SqliteSaver

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END

# Create checkpointer
memory = SqliteSaver.from_conn_string("./my_agent_checkpoints.db")

# Compile with checkpointing
app = workflow.compile(checkpointer=memory)
```

### Complete Persistence Example

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import time

class PersistentState(TypedDict):
    messages: Annotated[list, add]
    step_count: Annotated[int, add]
    user_id: str

llm = ChatOpenAI(model="gpt-4")

def slow_process(state: PersistentState) -> dict:
    """Simulate slow processing"""
    time.sleep(2)  # Simulate work
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": 1
    }

def another_step(state: PersistentState) -> dict:
    time.sleep(2)
    return {"step_count": 1}

# Build graph
workflow = StateGraph(PersistentState)
workflow.add_node("process", slow_process)
workflow.add_node("step2", another_step)

workflow.set_entry_point("process")
workflow.add_edge("process", "step2")
workflow.add_edge("step2", END)

# Compile with checkpointing
checkpointer = SqliteSaver.from_conn_string("./checkpoints.db")
app = workflow.compile(checkpointer=checkpointer)

# Run with thread_id for persistence
thread_id = "user-123-session-1"
config = {"configurable": {"thread_id": thread_id}}

print("Starting agent...")
result = app.invoke(
    {
        "messages": [HumanMessage(content="Hello")],
        "step_count": 0,
        "user_id": "user-123"
    },
    config=config
)

print(f"Completed with {result['step_count']} steps")

# Later: Resume the same conversation
print("\nResuming conversation...")
result2 = app.invoke(
    {
        "messages": [HumanMessage(content="Continue from where we left off")],
        "step_count": 0,  # Will be added to previous count
        "user_id": "user-123"
    },
    config=config  # Same thread_id
)

print(f"Total steps across sessions: {result2['step_count']}")
```

### Checkpoint Configuration

```python
# Different threads for different conversations
config_user1 = {"configurable": {"thread_id": "user-1-conversation-1"}}
config_user2 = {"configurable": {"thread_id": "user-2-conversation-1"}}

# Each gets independent state
app.invoke(state1, config=config_user1)
app.invoke(state2, config=config_user2)
```

---

## 🚨 Error Handling and Recovery

### Strategy 1: Error State Management

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class RobustState(TypedDict):
    data: str
    error: str | None
    retry_count: Annotated[int, add]
    max_retries: int

def risky_operation(state: RobustState) -> dict:
    """Operation that might fail"""
    try:
        # Simulate failure on first attempts
        if state["retry_count"] < 2:
            raise ValueError("Temporary failure")
        
        # Success
        return {
            "data": "Success!",
            "error": None
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "retry_count": 1
        }

def error_handler(state: RobustState) -> dict:
    """Handle errors"""
    print(f"Error occurred: {state['error']}")
    print(f"Retry attempt: {state['retry_count']}")
    return {}  # No updates, just logging

def should_retry(state: RobustState) -> str:
    """Decide whether to retry"""
    if state["error"] is None:
        return "success"
    
    if state["retry_count"] >= state["max_retries"]:
        return "failed"
    
    return "retry"

# Build graph with error handling
workflow = StateGraph(RobustState)

workflow.add_node("operation", risky_operation)
workflow.add_node("error_handler", error_handler)

workflow.set_entry_point("operation")

workflow.add_conditional_edges(
    "operation",
    should_retry,
    {
        "success": END,
        "retry": "error_handler",
        "failed": END
    }
)

workflow.add_edge("error_handler", "operation")  # Retry loop

app = workflow.compile()

# Test
result = app.invoke({
    "data": "",
    "error": None,
    "retry_count": 0,
    "max_retries": 3
})

print(f"Final result: {result}")
```

### Strategy 2: Try-Catch in Nodes

```python
def safe_node(state: AgentState) -> dict:
    """Node with comprehensive error handling"""
    try:
        # Main operation
        result = potentially_failing_operation(state["input"])
        
        return {
            "output": result,
            "status": "success"
        }
    
    except ValueError as e:
        # Handle specific error types
        return {
            "output": None,
            "status": "validation_error",
            "error_message": str(e)
        }
    
    except Exception as e:
        # Catch-all for unexpected errors
        return {
            "output": None,
            "status": "unexpected_error",
            "error_message": str(e)
        }
```

---

## 🔍 Debugging and Visualization

### 1. Print State at Each Step

```python
for step in app.stream(initial_state):
    print(f"\n{'='*60}")
    print(f"Current state: {step}")
```

### 2. Get Graph Structure

```python
# Get the graph as a Mermaid diagram
graph_diagram = app.get_graph().draw_mermaid()
print(graph_diagram)
```

**Example output:**
```
graph TD
    __start__ --> process
    process --> review
    review --> improve
    improve --> review
    review --> __end__
```

### 3. Inspect Checkpoints

```python
# Get all checkpoints for a thread
checkpoints = list(checkpointer.list(config))

for checkpoint in checkpoints:
    print(f"Step: {checkpoint}")
```

---

## 🏭 Production-Ready Pattern

Here's a complete, production-ready agent template:

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State definition
class ProductionState(TypedDict):
    messages: Annotated[list, add]
    user_id: str
    session_id: str
    iteration: Annotated[int, add]
    max_iterations: int
    status: Literal["processing", "success", "error"]
    error_message: str | None
    metadata: dict

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
checkpointer = SqliteSaver.from_conn_string("./production.db")

# Node implementations
def process_input(state: ProductionState) -> dict:
    """Main processing node"""
    try:
        logger.info(f"Processing for user: {state['user_id']}")
        
        response = llm.invoke(state["messages"])
        
        return {
            "messages": [response],
            "iteration": 1,
            "status": "processing"
        }
    
    except Exception as e:
        logger.error(f"Error in process_input: {e}")
        return {
            "status": "error",
            "error_message": str(e)
        }

def validate_output(state: ProductionState) -> dict:
    """Validation node"""
    try:
        # Add validation logic
        last_message = state["messages"][-1].content
        
        if len(last_message) < 10:
            return {
                "status": "error",
                "error_message": "Response too short"
            }
        
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Error in validate_output: {e}")
        return {
            "status": "error",
            "error_message": str(e)
        }

# Routing logic
def route_next(state: ProductionState) -> str:
    """Determine next step"""
    if state["status"] == "error":
        return "error"
    
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    
    if state["status"] == "success":
        return "end"
    
    return "validate"

# Build graph
workflow = StateGraph(ProductionState)

workflow.add_node("process", process_input)
workflow.add_node("validate", validate_output)

workflow.set_entry_point("process")

workflow.add_conditional_edges(
    "process",
    route_next,
    {
        "validate": "validate",
        "error": END,
        "end": END
    }
)

workflow.add_conditional_edges(
    "validate",
    route_next,
    {
        "end": END,
        "error": END
    }
)

# Compile with checkpointing
app = workflow.compile(checkpointer=checkpointer)

# Usage function
def run_agent(user_id: str, session_id: str, message: str):
    """Run the agent with proper configuration"""
    config = {
        "configurable": {
            "thread_id": f"{user_id}-{session_id}"
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "session_id": session_id,
        "iteration": 0,
        "max_iterations": 3,
        "status": "processing",
        "error_message": None,
        "metadata": {}
    }
    
    try:
        result = app.invoke(initial_state, config=config)
        
        if result["status"] == "error":
            logger.error(f"Agent failed: {result['error_message']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

# Example usage
if __name__ == "__main__":
    result = run_agent(
        user_id="user-123",
        session_id="session-456",
        message="Explain quantum computing"
    )
    
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Response: {result['messages'][-1].content}")
```

---

## ⚠️ Common Mistakes to Avoid

### Mistake 1: Not Using Reducers

```python
# ❌ Without reducer - messages get replaced
class BadState(TypedDict):
    messages: list  # No reducer!

# Returns [msg3] instead of [msg1, msg2, msg3]
```

```python
# ✅ With reducer - messages accumulate
from typing import Annotated
from operator import add

class GoodState(TypedDict):
    messages: Annotated[list, add]
```

### Mistake 2: Forgetting to Set Entry Point

```python
# ❌ No entry point
workflow.add_node("node1", func1)
app = workflow.compile()  # Error! No entry point

# ✅ Set entry point
workflow.set_entry_point("node1")
```

### Mistake 3: Disconnected Nodes

```python
# ❌ Node added but no edges
workflow.add_node("orphan", orphan_func)  # Never reached!

# ✅ Ensure all nodes are connected
workflow.add_edge("previous", "orphan")
```

### Mistake 4: Returning Full State Instead of Updates

```python
# ❌ Don't return the full state
def bad_node(state):
    state["new_field"] = "value"
    return state  # Returns everything

# ✅ Return only updates
def good_node(state):
    return {"new_field": "value"}  # Will be merged
```

---

## 🧠 Key Concepts to Remember

1. **StateGraph** for custom state, **MessageGraph** for simple chat
2. **Reducers** control how state updates are merged (add, replace, custom)
3. **Nodes** return **partial updates**, not full state
4. **Conditional edges** use functions to dynamically route
5. **Checkpointing** enables pause/resume and persistence
6. Always handle errors in nodes
7. Use `stream()` to debug step-by-step

---

## 🚀 What's Next?

In **Chapter 3**, we'll explore:
- The OODA Loop (Observe-Orient-Decide-Act)
- Agent loop design patterns
- Building perception layers
- Action execution frameworks
- Feedback loops and iteration strategies

---

## ✅ Chapter 2 Complete!

**You now understand:**
- ✅ StateGraph vs MessageGraph
- ✅ State schema design with reducers
- ✅ Node implementation best practices
- ✅ Fixed and conditional edges
- ✅ Compilation and execution modes
- ✅ Checkpointing and persistence
- ✅ Error handling strategies
- ✅ Production-ready patterns

**Ready for Chapter 3?** Just say "Continue to Chapter 3" or ask any questions about Chapter 2!