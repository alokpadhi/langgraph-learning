# Chapter 7: Tool Use & Function Calling (Production Patterns)

## 🎯 The Problem: Agents Need to Interact with the World

LLMs alone can only generate text. To be useful, agents need to:
- Search the web for current information
- Execute code and see results
- Query databases
- Call APIs
- Manipulate files
- Perform calculations

**Tools** bridge the gap between reasoning (LLM) and action (real world).

---

## 🔧 Part 1: Tool Fundamentals

### What is a Tool?

A **tool** is a function that an LLM can call to perform actions or retrieve information.

```python
from langchain_core.tools import tool

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    # Implementation here
    return f"Wikipedia results for: {query}"
```

**Key components:**
1. **Function signature** - defines inputs
2. **Docstring** - tells LLM what it does (critical!)
3. **Return value** - gives LLM information back
4. **Type hints** - ensures structured inputs

---

## 💻 Part 2: Defining Tools Properly

### Method 1: Using @tool Decorator (Recommended)

```python
from langchain_core.tools import tool
from typing import Optional

@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.
    
    Args:
        expression: A valid Python math expression like '2+2' or '15*23'
    
    Returns:
        The calculation result as a string
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def web_search(query: str, num_results: Optional[int] = 5) -> str:
    """
    Search the web for current information.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
    
    Returns:
        Formatted search results
    """
    # Simulated search
    return f"Found {num_results} results for '{query}'"

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """
    Get current weather for a city.
    
    Args:
        city: City name
        units: Temperature units - 'celsius' or 'fahrenheit' (default: celsius)
    
    Returns:
        Current weather information
    """
    # Simulated weather API
    weather_data = {
        "paris": {"temp": 18, "condition": "Partly cloudy"},
        "tokyo": {"temp": 22, "condition": "Sunny"},
        "new york": {"temp": 15, "condition": "Rainy"}
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        temp = data["temp"]
        if units == "fahrenheit":
            temp = (temp * 9/5) + 32
        
        return f"{city}: {temp}°{units[0].upper()}, {data['condition']}"
    
    return f"Weather data not available for {city}"
```

**Best practices for tool definitions:**
1. ✅ Clear, descriptive docstring (LLM reads this!)
2. ✅ Type hints for all parameters
3. ✅ Default values for optional parameters
4. ✅ Return strings (easier for LLM to process)
5. ✅ Handle errors gracefully

### Method 2: Using Pydantic for Complex Inputs

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

class WeatherQuery(BaseModel):
    """Input schema for weather queries"""
    city: str = Field(description="The city name")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature units"
    )
    include_forecast: bool = Field(
        default=False,
        description="Whether to include 3-day forecast"
    )

@tool(args_schema=WeatherQuery)
def get_weather_advanced(city: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """
    Get detailed weather information for a city.
    
    This tool provides current weather and optionally a 3-day forecast.
    """
    # Implementation
    return f"Weather for {city} in {units}"
```

---

## 🏭 Part 3: Production Tool Usage with LangGraph

### Complete Production Pattern

```python
from typing import TypedDict, Annotated, Sequence, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define tools
@tool
def search_database(query: str) -> str:
    """
    Search the product database for items.
    
    Args:
        query: Search term (product name, category, etc.)
    
    Returns:
        List of matching products
    """
    # Simulated database
    products = {
        "laptop": "MacBook Pro 14-inch - $1999",
        "phone": "iPhone 15 Pro - $999",
        "tablet": "iPad Air - $599"
    }
    
    query_lower = query.lower()
    results = [v for k, v in products.items() if query_lower in k]
    
    if results:
        return f"Found products: {', '.join(results)}"
    return f"No products found matching '{query}'"

@tool
def calculate_discount(price: float, discount_percent: float) -> str:
    """
    Calculate discounted price.
    
    Args:
        price: Original price
        discount_percent: Discount percentage (e.g., 20 for 20%)
    
    Returns:
        Final price after discount
    """
    try:
        final_price = price * (1 - discount_percent / 100)
        savings = price - final_price
        return f"Original: ${price:.2f}, Discount: {discount_percent}%, Final: ${final_price:.2f} (Save ${savings:.2f})"
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def check_inventory(product_name: str) -> str:
    """
    Check inventory status for a product.
    
    Args:
        product_name: Name of the product
    
    Returns:
        Inventory status
    """
    # Simulated inventory
    inventory = {
        "macbook": {"stock": 15, "warehouse": "NY"},
        "iphone": {"stock": 50, "warehouse": "CA"},
        "ipad": {"stock": 0, "warehouse": "TX"}
    }
    
    product_lower = product_name.lower()
    for key, value in inventory.items():
        if key in product_lower:
            stock = value["stock"]
            warehouse = value["warehouse"]
            
            if stock > 0:
                return f"{product_name}: {stock} units in stock at {warehouse} warehouse"
            else:
                return f"{product_name}: Out of stock"
    
    return f"Product '{product_name}' not found in inventory system"

# Tools list
tools = [search_database, calculate_discount, check_inventory]

# State definition
class ToolAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    tool_call_count: Annotated[int, add]
    max_tool_calls: int
    status: Literal["running", "completed", "error", "max_calls_reached"]

# Initialize LLM with tools
llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# System prompt
AGENT_SYSTEM_PROMPT = """You are a helpful shopping assistant with access to tools.

You can:
- Search for products in the database
- Calculate discounts
- Check inventory status

Use tools when needed to provide accurate information. Always explain what you're doing."""

# Node: Agent with tools
def agent_node(state: ToolAgentState) -> dict:
    """Agent decides whether to use tools or respond"""
    
    try:
        logger.info(f"Agent processing (tool calls so far: {state['tool_call_count']})")
        
        # Check if we've hit max tool calls
        if state["tool_call_count"] >= state["max_tool_calls"]:
            logger.warning("Max tool calls reached")
            return {
                "messages": [AIMessage(content="I've reached my tool usage limit. Please start a new conversation.")],
                "status": "max_calls_reached"
            }
        
        # Prepare messages with system prompt
        system_msg = {"type": "system", "content": AGENT_SYSTEM_PROMPT}
        messages = [system_msg] + list(state["messages"])
        
        # Invoke LLM
        response = llm_with_tools.invoke(messages)
        
        # Count tool calls in response
        tool_calls_in_response = 0
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls_in_response = len(response.tool_calls)
            logger.info(f"Agent requested {tool_calls_in_response} tool call(s)")
        
        return {
            "messages": [response],
            "tool_call_count": tool_calls_in_response,
            "status": "running"
        }
    
    except Exception as e:
        logger.error(f"Agent node error: {e}")
        return {
            "messages": [AIMessage(content=f"I encountered an error: {str(e)}")],
            "status": "error"
        }

# Create tool node
tool_node = ToolNode(tools)

# Router function
def should_continue(state: ToolAgentState) -> str:
    """Decide next step based on last message"""
    
    # Check status
    if state["status"] in ["error", "max_calls_reached"]:
        return "end"
    
    # Check last message for tool calls
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("Routing to tools")
        return "tools"
    
    logger.info("Agent finished - no more tool calls")
    return "end"

# Build graph
workflow = StateGraph(ToolAgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# After tools, always go back to agent
workflow.add_edge("tools", "agent")

# Compile with checkpointing
checkpointer = SqliteSaver.from_conn_string("./tool_agent.db")
tool_agent = workflow.compile(checkpointer=checkpointer)

# Production API
def chat_with_tools(
    message: str,
    session_id: str = "default",
    max_tool_calls: int = 10
) -> dict:
    """
    Chat with tool-using agent.
    
    Args:
        message: User message
        session_id: Session identifier for conversation continuity
        max_tool_calls: Maximum number of tool calls allowed
    
    Returns:
        dict with response and metadata
    """
    
    config = {
        "configurable": {
            "thread_id": f"tool-chat-{session_id}"
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "tool_call_count": 0,
        "max_tool_calls": max_tool_calls,
        "status": "running"
    }
    
    try:
        result = tool_agent.invoke(initial_state, config=config)
        
        # Extract final response
        final_message = result["messages"][-1]
        
        # Count tool uses
        tool_messages = [
            msg for msg in result["messages"] 
            if isinstance(msg, ToolMessage)
        ]
        
        return {
            "success": True,
            "response": final_message.content if hasattr(final_message, 'content') else str(final_message),
            "tool_calls_used": result["tool_call_count"],
            "status": result["status"],
            "tools_used": [msg.name for msg in tool_messages] if tool_messages else []
        }
    
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    session = "user-123"
    
    queries = [
        "What laptops do you have?",
        "What's the price if I get a 15% discount on the MacBook?",
        "Is the MacBook in stock?"
    ]
    
    print("\n" + "="*60)
    print("TOOL-USING AGENT DEMO")
    print("="*60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"User: {query}")
        
        result = chat_with_tools(query, session_id=session)
        
        if result["success"]:
            print(f"Agent: {result['response']}")
            print(f"Tools used: {result['tools_used']}")
            print(f"Total tool calls: {result['tool_calls_used']}")
        else:
            print(f"Error: {result['error']}")
```

---

## 🔄 Part 4: Parallel Tool Execution

### The Problem: Sequential vs Parallel

```python
# Sequential (slow):
result1 = search_database("laptop")
result2 = check_inventory("laptop")
result3 = calculate_discount(1999, 20)
# Total time = T1 + T2 + T3

# Parallel (fast):
results = execute_in_parallel([
    search_database("laptop"),
    check_inventory("laptop"),
    calculate_discount(1999, 20)
])
# Total time = max(T1, T2, T3)
```

### LangGraph's Automatic Parallel Execution

**Good news:** LangGraph's `ToolNode` automatically executes independent tool calls in parallel!

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

@tool
def fetch_user_data(user_id: str) -> str:
    """Fetch user profile data."""
    import time
    time.sleep(2)  # Simulate API call
    return f"User {user_id}: John Doe, Premium Member"

@tool
def fetch_order_history(user_id: str) -> str:
    """Fetch user's order history."""
    import time
    time.sleep(2)  # Simulate API call
    return f"Orders for {user_id}: 3 recent orders"

@tool
def fetch_recommendations(user_id: str) -> str:
    """Get product recommendations."""
    import time
    time.sleep(2)  # Simulate API call
    return f"Recommendations for {user_id}: Laptop, Phone, Tablet"

tools = [fetch_user_data, fetch_order_history, fetch_recommendations]

# When LLM requests multiple tools at once, ToolNode runs them in parallel
tool_node = ToolNode(tools)

# Example: If LLM calls all 3 tools together, they run in parallel
# Total time: ~2 seconds instead of ~6 seconds
```

### Demonstrating Parallel Execution

```python
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
import time

# Tools that simulate API delays
@tool
def api_call_1(param: str) -> str:
    """First API call."""
    time.sleep(2)
    return f"API 1 result: {param}"

@tool
def api_call_2(param: str) -> str:
    """Second API call."""
    time.sleep(2)
    return f"API 2 result: {param}"

@tool
def api_call_3(param: str) -> str:
    """Third API call."""
    time.sleep(2)
    return f"API 3 result: {param}"

tools = [api_call_1, api_call_2, api_call_3]

# State
class ParallelState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]

# LLM with tools
llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent(state: ParallelState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)  # Automatically parallel!

def should_continue(state: ParallelState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"

# Build graph
workflow = StateGraph(ParallelState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})
workflow.add_edge("tools", "agent")

parallel_agent = workflow.compile()

# Test
start = time.time()
result = parallel_agent.invoke({
    "messages": [HumanMessage(content="Call all three APIs with parameter 'test'")]
})
elapsed = time.time() - start

print(f"\nTotal execution time: {elapsed:.2f}s")
print("(Should be ~2s if parallel, ~6s if sequential)")
```

---

## 🛡️ Part 5: Error Handling & Retries

### Built-in Tool Error Handling

```python
from langchain_core.tools import tool
from typing import Optional

@tool
def robust_api_call(endpoint: str, retries: int = 3) -> str:
    """
    Call an API with automatic retry logic.
    
    Args:
        endpoint: API endpoint to call
        retries: Number of retry attempts (default: 3)
    
    Returns:
        API response or error message
    """
    import random
    
    for attempt in range(retries):
        try:
            # Simulate API call that might fail
            if random.random() < 0.3:  # 30% failure rate
                raise Exception("API temporarily unavailable")
            
            return f"Success: Data from {endpoint}"
        
        except Exception as e:
            if attempt < retries - 1:
                # Wait before retry (exponential backoff)
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                # Final attempt failed
                return f"Error after {retries} attempts: {str(e)}"
    
    return "Unexpected error"
```

### Custom Tool Node with Error Recovery

```python
from typing import TypedDict, Annotated, Sequence, Any
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
import logging

logger = logging.getLogger(__name__)

# State with error tracking
class ErrorHandlingState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    errors: Annotated[list[str], add]
    retry_count: Annotated[int, add]
    max_retries: int

# Tool that might fail
@tool
def unreliable_tool(query: str) -> str:
    """A tool that sometimes fails."""
    import random
    
    if random.random() < 0.5:  # 50% failure rate
        raise ValueError("Tool execution failed")
    
    return f"Successfully processed: {query}"

tools = [unreliable_tool]

# Custom tool execution with error handling
def execute_tools_with_retry(state: ErrorHandlingState) -> dict:
    """Execute tools with error handling and retry logic"""
    
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {}
    
    tool_messages = []
    errors = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Find the tool
        tool_fn = None
        for t in tools:
            if t.name == tool_name:
                tool_fn = t
                break
        
        if not tool_fn:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(error_msg)
            tool_messages.append(
                ToolMessage(content=error_msg, tool_call_id=tool_id)
            )
            errors.append(error_msg)
            continue
        
        # Execute with retry
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = tool_fn.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )
                logger.info(f"Tool {tool_name} succeeded on attempt {attempt + 1}")
                break
            
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Tool {tool_name} failed (attempt {attempt + 1}): {e}")
                    continue
                else:
                    error_msg = f"Tool {tool_name} failed after {max_attempts} attempts: {str(e)}"
                    logger.error(error_msg)
                    tool_messages.append(
                        ToolMessage(content=error_msg, tool_call_id=tool_id)
                    )
                    errors.append(error_msg)
    
    return {
        "messages": tool_messages,
        "errors": errors,
        "retry_count": 1 if errors else 0
    }

# Build agent with custom tool handler
llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent(state: ErrorHandlingState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: ErrorHandlingState):
    last_message = state["messages"][-1]
    
    # Check for max retries
    if state["retry_count"] >= state["max_retries"]:
        return "end"
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    return "end"

workflow = StateGraph(ErrorHandlingState)
workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tools_with_retry)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})
workflow.add_edge("tools", "agent")

error_handling_agent = workflow.compile()
```

---

## 🎨 Part 6: Custom Tool Creation Patterns

### Pattern 1: Database Tool

```python
from langchain_core.tools import tool
from typing import List, Dict, Any
import json

@tool
def query_database(sql: str) -> str:
    """
    Execute SQL query on the database.
    
    Args:
        sql: SQL query to execute (SELECT only for safety)
    
    Returns:
        Query results as JSON string
    """
    # Simulated database
    mock_db = {
        "users": [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25}
        ],
        "orders": [
            {"id": 101, "user_id": 1, "amount": 50.00},
            {"id": 102, "user_id": 2, "amount": 75.50}
        ]
    }
    
    # Very basic SQL parsing (in production, use proper SQL library)
    sql_lower = sql.lower().strip()
    
    if not sql_lower.startswith("select"):
        return "Error: Only SELECT queries are allowed"
    
    # Extract table name (very simplified)
    if "from users" in sql_lower:
        results = mock_db["users"]
    elif "from orders" in sql_lower:
        results = mock_db["orders"]
    else:
        return "Error: Table not found"
    
    return json.dumps(results, indent=2)
```

### Pattern 2: API Tool with Authentication

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import os

class APIConfig(BaseModel):
    """Configuration for API tool"""
    base_url: str = Field(default="https://api.example.com")
    api_key: str = Field(default_factory=lambda: os.getenv("API_KEY", ""))

@tool
def call_external_api(endpoint: str, method: str = "GET") -> str:
    """
    Call external API with authentication.
    
    Args:
        endpoint: API endpoint path (e.g., '/users/123')
        method: HTTP method (GET, POST, etc.)
    
    Returns:
        API response
    """
    config = APIConfig()
    
    if not config.api_key:
        return "Error: API key not configured"
    
    # Simulate API call
    full_url = f"{config.base_url}{endpoint}"
    
    # In production, use requests library:
    # response = requests.request(
    #     method=method,
    #     url=full_url,
    #     headers={"Authorization": f"Bearer {config.api_key}"}
    # )
    # return response.text
    
    return f"Simulated {method} request to {full_url}"
```

### Pattern 3: File System Tool

```python
from langchain_core.tools import tool
import os
from pathlib import Path

@tool
def read_file(filepath: str) -> str:
    """
    Read contents of a file safely.
    
    Args:
        filepath: Path to the file to read
    
    Returns:
        File contents or error message
    """
    try:
        # Security: Only allow reading from specific directory
        safe_dir = Path("./allowed_files")
        safe_dir.mkdir(exist_ok=True)
        
        file_path = (safe_dir / filepath).resolve()
        
        # Ensure path is within safe directory
        if not str(file_path).startswith(str(safe_dir.resolve())):
            return "Error: Access denied - path outside allowed directory"
        
        if not file_path.exists():
            return f"Error: File '{filepath}' not found"
        
        # Limit file size (prevent reading huge files)
        max_size = 1024 * 1024  # 1MB
        if file_path.stat().st_size > max_size:
            return "Error: File too large (max 1MB)"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def list_files(directory: str = ".") -> str:
    """
    List files in a directory.
    
    Args:
        directory: Directory path (default: current directory)
    
    Returns:
        List of files
    """
    try:
        safe_dir = Path("./allowed_files")
        target_dir = (safe_dir / directory).resolve()
        
        # Security check
        if not str(target_dir).startswith(str(safe_dir.resolve())):
            return "Error: Access denied"
        
        if not target_dir.exists():
            return f"Error: Directory '{directory}' not found"
        
        files = [f.name for f in target_dir.iterdir() if f.is_file()]
        
        return f"Files in '{directory}': {', '.join(files) if files else 'No files found'}"
    
    except Exception as e:
        return f"Error listing files: {str(e)}"
```

---

## 📡 Part 7: Streaming with Tools

### Streaming Tool Calls and Results

```python
from typing import TypedDict, Annotated, Sequence, AsyncIterator
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
import asyncio

# Tools
@tool
def slow_operation(task: str) -> str:
    """Perform a slow operation."""
    import time
    time.sleep(2)
    return f"Completed: {task}"

tools = [slow_operation]

# State
class StreamState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]

# LLM
llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent(state: StreamState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: StreamState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"

workflow = StateGraph(StreamState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})
workflow.add_edge("tools", "agent")

streaming_agent = workflow.compile()

# Stream events
async def stream_agent_response(message: str):
    """Stream agent execution events"""
    
    print(f"\n{'='*60}")
    print(f"USER: {message}")
    print(f"{'='*60}\n")
    
    async for event in streaming_agent.astream({
        "messages": [HumanMessage(content=message)]
    }):
        for node_name, node_output in event.items():
            print(f"📍 Node: {node_name}")
            
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    msg_type = type(msg).__name__
                    
                    if hasattr(msg, 'content'):
                        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        print(f"   {msg_type}: {content}")
                    
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"   🔧 Tool Call: {tc['name']}({tc['args']})")
            
            print()

# Run streaming example
if __name__ == "__main__":
    asyncio.run(stream_agent_response(
        "Please perform a slow operation on task 'data processing'"
    ))
```

---

## 🏆 Part 8: Production Best Practices Summary

### 1. Tool Definition Checklist

```python
from langchain_core.tools import tool
from typing import Optional

@tool
def production_ready_tool(
    required_param: str,
    optional_param: Optional[int] = None
) -> str:
    """
    ✅ Clear description of what the tool does
    ✅ When to use this tool
    
    Args:
        required_param: ✅ Description of parameter
        optional_param: ✅ Description with default value
    
    Returns:
        ✅ Description of return value
    """
    try:
        # ✅ Validate inputs
        if not required_param:
            return "Error: required_param cannot be empty"
        
        # ✅ Perform operation
        result = f"Processed: {required_param}"
        
        # ✅ Return string (easier for LLM)
        return result
    
    except Exception as e:
        # ✅ Handle errors gracefully
        return f"Error: {str(e)}"
```

### 2. Security Considerations

```python
# ✅ Whitelist allowed operations
ALLOWED_OPERATIONS = ["search", "calculate", "translate"]

@tool
def secure_tool(operation: str, data: str) -> str:
    """Execute operation securely."""
    
    # Validate operation
    if operation not in ALLOWED_OPERATIONS:
        return f"Error: Operation '{operation}' not allowed"
    
    # Sanitize inputs
    data = data.strip()[:1000]  # Limit length
    
    # Execute safely
    # ...
    
    return result

# ✅ Rate limiting
from functools import wraps
import time

def rate_limit(max_calls: int, period: int):
    """Rate limit decorator"""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls
            calls[:] = [c for c in calls if c > now - period]
            
            if len(calls) >= max_calls:
                return f"Error: Rate limit exceeded ({max_calls} calls per {period}s)"
            
            calls.append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

@tool
@rate_limit(max_calls=10, period=60)
def rate_limited_api(query: str) -> str:
    """API with rate limiting."""
    return f"Result for: {query}"
```

### 3. Logging and Monitoring

```python
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def logged_tool(func):
    """Decorator to log tool execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Tool {func.__name__} called with args={args}, kwargs={kwargs}")
        
        try:
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            
            logger.info(f"Tool {func.__name__} completed in {elapsed:.2f}s")
            return result
        
        except Exception as e:
            logger.error(f"Tool {func.__name__} failed: {e}")
            raise
    
    return wrapper

@tool
@logged_tool
def monitored_tool(param: str) -> str:
    """Tool with logging."""
    return f"Result: {param}"
```

---

## 🧠 Key Concepts to Remember

1. **Use @tool decorator** for clean tool definition
2. **Docstrings are critical** - LLM reads them to understand usage
3. **bind_tools + ToolNode** is the production pattern
4. **ToolNode automatically parallelizes** independent tool calls
5. **Always handle errors** in tool implementations
6. **Return strings** for easier LLM processing
7. **Validate and sanitize inputs** for security
8. **Use type hints** and Pydantic for complex inputs
9. **Stream events** for better UX in long-running tasks
10. **Log tool usage** for debugging and monitoring

---

## 🚀 What's Next?

In **Chapter 8**, we'll explore:
- **Model Context Protocol (MCP)** - standardized tool interfaces
- MCP architecture and components
- Integrating MCP servers with LangGraph
- Building custom MCP tools
- Production MCP patterns

---

## ✅ Chapter 7 Complete!

**You now understand:**
- ✅ Proper tool definition with @tool decorator
- ✅ Production patterns with bind_tools and ToolNode
- ✅ Automatic parallel tool execution
- ✅ Error handling and retry logic
- ✅ Custom tool creation patterns (DB, API, files)
- ✅ Streaming with tools
- ✅ Security, rate limiting, and monitoring
- ✅ Complete production-ready tool agent

**Ready for Chapter 8?** Just say "Continue to Chapter 8" or ask any questions!