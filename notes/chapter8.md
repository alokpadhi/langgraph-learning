# Chapter 8: Model Context Protocol (MCP)

## 🎯 The Problem: Tools Need Standardization

So far, we've been creating tools ad-hoc:

```python
@tool
def my_tool():
    """My custom tool"""
    pass
```

**Problems with this approach:**
- ❌ Each tool is defined differently
- ❌ No standardized way to discover available tools
- ❌ Hard to share tools across applications
- ❌ No separation between tool logic and LLM integration
- ❌ Difficult to version and manage tools at scale

**MCP (Model Context Protocol)** solves this by providing a **standardized protocol** for exposing tools, resources, and prompts to LLMs.

---

## 🏗️ What is MCP?

**Model Context Protocol** is an open protocol developed by Anthropic that standardizes how AI applications connect to external data sources and tools.

### Key Concepts

```
┌─────────────────────────────────────────────────┐
│                 MCP ARCHITECTURE                 │
│                                                  │
│  ┌──────────┐         ┌──────────┐             │
│  │   MCP    │ ◄─────► │   MCP    │             │
│  │  Client  │  JSON   │  Server  │             │
│  │ (Agent)  │   RPC   │ (Tools)  │             │
│  └──────────┘         └──────────┘             │
│       ↑                     ↓                    │
│       │              ┌──────────────┐           │
│       │              │ - Tools      │           │
│       │              │ - Resources  │           │
│       └──────────────│ - Prompts    │           │
│                      └──────────────┘           │
└─────────────────────────────────────────────────┘
```

### MCP Components

1. **MCP Server**: Exposes tools, resources, and prompts
2. **MCP Client**: Connects to servers and uses their capabilities
3. **Transport Layer**: Communication protocol (stdio, HTTP, WebSocket)
4. **Tools**: Functions the LLM can call
5. **Resources**: Data the LLM can access (files, databases, etc.)
6. **Prompts**: Reusable prompt templates

---

## 📦 Part 1: Installing MCP

```bash
# Install MCP SDK
pip install mcp

# Or with uv
uv pip install mcp

# For additional integrations
pip install mcp[cli]  # CLI tools
pip install httpx     # For HTTP transport
```

---

## 🎓 Part 2: Understanding MCP Servers (Educational)

### Simple MCP Server Example

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio

# Create MCP server
app = Server("demo-server")

# Define a tool
@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        )
    ]

# Implement tool
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "get_weather":
        city = arguments.get("city", "")
        
        # Simulated weather data
        weather_data = {
            "paris": "18°C, Partly cloudy",
            "tokyo": "22°C, Sunny",
            "new york": "15°C, Rainy"
        }
        
        result = weather_data.get(city.lower(), f"Weather data not available for {city}")
        
        return [TextContent(
            type="text",
            text=result
        )]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

# Run server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

**This MCP server:**
- ✅ Exposes a `get_weather` tool
- ✅ Uses stdio transport (standard input/output)
- ✅ Can be discovered by any MCP client
- ✅ Follows MCP protocol standards

---

## 🏭 Part 3: Production MCP Server

### Complete Production MCP Server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource, Prompt, PromptMessage
from typing import Any
import asyncio
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create MCP server
app = Server("production-mcp-server")

# ==================== TOOLS ====================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools"""
    logger.info("Listing tools")
    
    return [
        Tool(
            name="search_database",
            description="Search the product database for items. Use this to find products by name or category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (product name or category)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="calculate_price",
            description="Calculate final price with tax and discounts",
            inputSchema={
                "type": "object",
                "properties": {
                    "base_price": {
                        "type": "number",
                        "description": "Base price in dollars"
                    },
                    "discount_percent": {
                        "type": "number",
                        "description": "Discount percentage (0-100)",
                        "default": 0
                    },
                    "tax_rate": {
                        "type": "number",
                        "description": "Tax rate (0-1)",
                        "default": 0.08
                    }
                },
                "required": ["base_price"]
            }
        ),
        Tool(
            name="check_availability",
            description="Check product availability and stock levels",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product identifier"
                    },
                    "warehouse": {
                        "type": "string",
                        "description": "Warehouse location (NY, CA, TX)",
                        "default": "NY"
                    }
                },
                "required": ["product_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute tool calls"""
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        if name == "search_database":
            result = await search_database_impl(
                arguments.get("query", ""),
                arguments.get("limit", 10)
            )
            
        elif name == "calculate_price":
            result = await calculate_price_impl(
                arguments.get("base_price", 0),
                arguments.get("discount_percent", 0),
                arguments.get("tax_rate", 0.08)
            )
            
        elif name == "check_availability":
            result = await check_availability_impl(
                arguments.get("product_id", ""),
                arguments.get("warehouse", "NY")
            )
        
        else:
            result = f"Unknown tool: {name}"
        
        return [TextContent(type="text", text=result)]
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]

# Tool implementations
async def search_database_impl(query: str, limit: int) -> str:
    """Search product database"""
    # Simulated database
    products = [
        {"id": "p1", "name": "Laptop Pro", "category": "electronics", "price": 1299},
        {"id": "p2", "name": "Wireless Mouse", "category": "electronics", "price": 29},
        {"id": "p3", "name": "Office Chair", "category": "furniture", "price": 199},
        {"id": "p4", "name": "Desk Lamp", "category": "furniture", "price": 45},
    ]
    
    query_lower = query.lower()
    results = [
        p for p in products 
        if query_lower in p["name"].lower() or query_lower in p["category"].lower()
    ]
    
    results = results[:limit]
    
    if not results:
        return f"No products found matching '{query}'"
    
    return json.dumps(results, indent=2)

async def calculate_price_impl(base_price: float, discount_percent: float, tax_rate: float) -> str:
    """Calculate final price"""
    if base_price < 0:
        return "Error: Base price cannot be negative"
    
    if not 0 <= discount_percent <= 100:
        return "Error: Discount must be between 0 and 100"
    
    if not 0 <= tax_rate <= 1:
        return "Error: Tax rate must be between 0 and 1"
    
    # Calculate
    discount_amount = base_price * (discount_percent / 100)
    price_after_discount = base_price - discount_amount
    tax_amount = price_after_discount * tax_rate
    final_price = price_after_discount + tax_amount
    
    return json.dumps({
        "base_price": base_price,
        "discount_percent": discount_percent,
        "discount_amount": round(discount_amount, 2),
        "price_after_discount": round(price_after_discount, 2),
        "tax_rate": tax_rate,
        "tax_amount": round(tax_amount, 2),
        "final_price": round(final_price, 2)
    }, indent=2)

async def check_availability_impl(product_id: str, warehouse: str) -> str:
    """Check product availability"""
    # Simulated inventory
    inventory = {
        "p1": {"NY": 15, "CA": 8, "TX": 20},
        "p2": {"NY": 100, "CA": 75, "TX": 50},
        "p3": {"NY": 5, "CA": 12, "TX": 0},
        "p4": {"NY": 30, "CA": 25, "TX": 15},
    }
    
    if product_id not in inventory:
        return f"Product '{product_id}' not found"
    
    if warehouse not in inventory[product_id]:
        return f"Warehouse '{warehouse}' not found"
    
    stock = inventory[product_id][warehouse]
    
    result = {
        "product_id": product_id,
        "warehouse": warehouse,
        "stock": stock,
        "status": "in_stock" if stock > 0 else "out_of_stock"
    }
    
    return json.dumps(result, indent=2)

# ==================== RESOURCES ====================

@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources"""
    logger.info("Listing resources")
    
    return [
        Resource(
            uri="catalog://products",
            name="Product Catalog",
            description="Complete product catalog with prices and descriptions",
            mimeType="application/json"
        ),
        Resource(
            uri="catalog://categories",
            name="Product Categories",
            description="List of all product categories",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content"""
    logger.info(f"Reading resource: {uri}")
    
    if uri == "catalog://products":
        products = [
            {"id": "p1", "name": "Laptop Pro", "category": "electronics", "price": 1299},
            {"id": "p2", "name": "Wireless Mouse", "category": "electronics", "price": 29},
            {"id": "p3", "name": "Office Chair", "category": "furniture", "price": 199},
            {"id": "p4", "name": "Desk Lamp", "category": "furniture", "price": 45},
        ]
        return json.dumps(products, indent=2)
    
    elif uri == "catalog://categories":
        categories = ["electronics", "furniture", "office supplies", "accessories"]
        return json.dumps(categories, indent=2)
    
    else:
        return f"Resource not found: {uri}"

# ==================== PROMPTS ====================

@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompt templates"""
    logger.info("Listing prompts")
    
    return [
        Prompt(
            name="product_recommendation",
            description="Generate product recommendations based on user preferences",
            arguments=[
                {
                    "name": "budget",
                    "description": "User's budget",
                    "required": True
                },
                {
                    "name": "category",
                    "description": "Preferred product category",
                    "required": False
                }
            ]
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> list[PromptMessage]:
    """Get prompt template with arguments"""
    logger.info(f"Getting prompt: {name} with arguments: {arguments}")
    
    if name == "product_recommendation":
        budget = arguments.get("budget", "unknown")
        category = arguments.get("category", "any category")
        
        prompt_text = f"""You are a helpful shopping assistant.

The user has a budget of {budget} and is interested in {category}.

Please:
1. Use the search_database tool to find products
2. Filter by the user's budget
3. Recommend 2-3 products that best match their needs
4. Explain why each product is a good choice

Be friendly and helpful in your recommendations."""
        
        return [
            PromptMessage(
                role="user",
                content={"type": "text", "text": prompt_text}
            )
        ]
    
    return []

# ==================== SERVER RUNNER ====================

async def run_server():
    """Run the MCP server"""
    logger.info("Starting production MCP server...")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
```

**Save this as `mcp_server.py`**

---

## 🔌 Part 4: Integrating MCP with LangGraph

### MCP Client for LangGraph

```python
from typing import TypedDict, Annotated, Sequence, Any
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State
class MCPAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    mcp_tools_available: bool
    tool_call_count: Annotated[int, add]

# ==================== MCP CLIENT ====================

class MCPToolClient:
    """Client for connecting to MCP servers"""
    
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self.session: ClientSession = None
        self.tools: list[dict] = []
        self._initialized = False
    
    async def __aenter__(self):
        """Initialize MCP connection"""
        logger.info(f"Connecting to MCP server: {self.server_script_path}")
        
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path],
            env=None
        )
        
        # Create stdio client context
        self.stdio_context = stdio_client(server_params)
        self.stdio_transport = await self.stdio_context.__aenter__()
        
        read, write = self.stdio_transport
        
        # Create session
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        
        # Initialize session
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        self.tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in response.tools
        ]
        
        logger.info(f"Connected to MCP server. Available tools: {[t['name'] for t in self.tools]}")
        self._initialized = True
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup MCP connection"""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        
        if self.stdio_context:
            await self.stdio_context.__aexit__(exc_type, exc_val, exc_tb)
        
        logger.info("Disconnected from MCP server")
    
    def get_langchain_tools(self) -> list[Tool]:
        """Convert MCP tools to LangChain Tool format"""
        langchain_tools = []
        
        for tool_info in self.tools:
            # Create a closure to capture tool_info
            def make_tool_func(tool_name: str):
                async def tool_func(**kwargs) -> str:
                    """Execute MCP tool"""
                    try:
                        result = await self.session.call_tool(tool_name, kwargs)
                        
                        # Extract text content from result
                        if result.content:
                            texts = [
                                c.text for c in result.content 
                                if hasattr(c, 'text')
                            ]
                            return "\n".join(texts) if texts else "No result"
                        
                        return "No result"
                    
                    except Exception as e:
                        logger.error(f"MCP tool error: {e}")
                        return f"Error: {str(e)}"
                
                return tool_func
            
            # Create LangChain Tool
            tool = Tool(
                name=tool_info["name"],
                description=tool_info["description"],
                func=make_tool_func(tool_info["name"]),
                coroutine=make_tool_func(tool_info["name"])  # For async
            )
            
            langchain_tools.append(tool)
        
        return langchain_tools

# ==================== LANGGRAPH AGENT WITH MCP ====================

async def create_mcp_agent(mcp_server_path: str):
    """Create LangGraph agent with MCP tools"""
    
    # Initialize MCP client
    mcp_client = MCPToolClient(mcp_server_path)
    await mcp_client.__aenter__()
    
    # Get tools from MCP server
    tools = mcp_client.get_langchain_tools()
    
    logger.info(f"Loaded {len(tools)} tools from MCP server")
    
    # Initialize LLM with tools
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    # Bind tools to LLM
    # Note: For async tools, we need to handle differently
    # LangChain's bind_tools expects sync functions
    # So we'll create a custom tool execution node
    
    # Agent node
    async def agent_node(state: MCPAgentState) -> dict:
        """Agent with MCP tools"""
        try:
            logger.info("Agent processing...")
            
            # Prepare tool descriptions for LLM
            tool_descriptions = "\n".join([
                f"- {t.name}: {t.description}"
                for t in tools
            ])
            
            system_prompt = f"""You are a helpful assistant with access to these tools:

{tool_descriptions}

To use a tool, respond in this exact format:
TOOL: tool_name
ARGS: {{"arg1": "value1", "arg2": "value2"}}

If you don't need a tool, just respond normally."""
            
            # Add system prompt to messages
            messages = [
                {"role": "system", "content": system_prompt}
            ] + [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant", 
                 "content": m.content}
                for m in state["messages"]
            ]
            
            response = llm.invoke(messages)
            response_text = response.content
            
            # Parse tool call from response
            if "TOOL:" in response_text and "ARGS:" in response_text:
                # Extract tool call
                lines = response_text.split('\n')
                tool_name = None
                tool_args = {}
                
                for line in lines:
                    if line.startswith("TOOL:"):
                        tool_name = line.replace("TOOL:", "").strip()
                    elif line.startswith("ARGS:"):
                        args_str = line.replace("ARGS:", "").strip()
                        try:
                            tool_args = json.loads(args_str)
                        except:
                            pass
                
                if tool_name:
                    # Find and execute tool
                    tool_func = None
                    for t in tools:
                        if t.name == tool_name:
                            tool_func = t.coroutine
                            break
                    
                    if tool_func:
                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                        
                        try:
                            result = await tool_func(**tool_args)
                            
                            return {
                                "messages": [
                                    AIMessage(content=f"Used tool {tool_name}"),
                                    ToolMessage(content=str(result), tool_call_id="mcp-call")
                                ],
                                "tool_call_count": 1
                            }
                        
                        except Exception as e:
                            logger.error(f"Tool execution error: {e}")
                            return {
                                "messages": [AIMessage(content=f"Tool error: {str(e)}")],
                                "tool_call_count": 1
                            }
            
            # No tool call, just respond
            return {
                "messages": [AIMessage(content=response_text)],
            }
        
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "messages": [AIMessage(content=f"Error: {str(e)}")]
            }
    
    # Router
    def should_continue(state: MCPAgentState) -> str:
        """Check if we should continue"""
        last_message = state["messages"][-1]
        
        # If last message is a tool result, go back to agent
        if isinstance(last_message, ToolMessage):
            return "agent"
        
        # Otherwise end
        return "end"
    
    # Build graph
    workflow = StateGraph(MCPAgentState)
    workflow.add_node("agent", agent_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "agent": "agent",
            "end": END
        }
    )
    
    # Compile
    checkpointer = SqliteSaver.from_conn_string("./mcp_agent.db")
    agent = workflow.compile(checkpointer=checkpointer)
    
    return agent, mcp_client

# ==================== USAGE ====================

async def chat_with_mcp_agent(message: str, mcp_server_path: str):
    """Chat with MCP-powered agent"""
    
    # Create agent with MCP tools
    agent, mcp_client = await create_mcp_agent(mcp_server_path)
    
    try:
        config = {
            "configurable": {
                "thread_id": "mcp-session-1"
            }
        }
        
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "mcp_tools_available": True,
            "tool_call_count": 0
        }
        
        print(f"\n{'='*60}")
        print(f"USER: {message}")
        print(f"{'='*60}\n")
        
        result = await agent.ainvoke(initial_state, config=config)
        
        # Print conversation
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                print(f"👤 User: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"🤖 Agent: {msg.content}")
            elif isinstance(msg, ToolMessage):
                print(f"🔧 Tool Result: {msg.content[:200]}...")
        
        print(f"\nTool calls made: {result['tool_call_count']}")
    
    finally:
        # Cleanup MCP client
        await mcp_client.__aexit__(None, None, None)

# Run example
if __name__ == "__main__":
    # First, make sure your mcp_server.py is saved
    mcp_server_path = "./mcp_server.py"
    
    queries = [
        "Search for electronics products",
        "Calculate the price of a $1299 laptop with 15% discount",
        "Check if product p1 is available in the CA warehouse"
    ]
    
    for query in queries:
        asyncio.run(chat_with_mcp_agent(query, mcp_server_path))
        print("\n" + "="*60 + "\n")
```

---

## 🎯 Part 5: Production MCP Integration with Proper bind_tools

### Better Integration Using MCP Native Support

```python
from typing import TypedDict, Annotated, Sequence
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== MCP TOOL WRAPPER ====================

class MCPToolWrapper:
    """Wrapper to convert MCP tools to LangChain tools with bind_tools support"""
    
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.session = None
        self.tools_info = []
        self.langchain_tools = []
    
    async def initialize(self):
        """Initialize connection to MCP server"""
        logger.info("Initializing MCP connection...")
        
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script],
            env=None
        )
        
        # Connect
        self.stdio_context = stdio_client(server_params)
        self.stdio_transport = await self.stdio_context.__aenter__()
        read, write = self.stdio_transport
        
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()
        
        # Get tools
        response = await self.session.list_tools()
        self.tools_info = response.tools
        
        # Convert to LangChain tools
        for tool_info in self.tools_info:
            lc_tool = self._create_langchain_tool(tool_info)
            self.langchain_tools.append(lc_tool)
        
        logger.info(f"Initialized {len(self.langchain_tools)} MCP tools")
    
    def _create_langchain_tool(self, mcp_tool):
        """Convert MCP tool to LangChain StructuredTool"""
        
        # Create Pydantic model for input schema
        schema_props = mcp_tool.inputSchema.get("properties", {})
        required_fields = mcp_tool.inputSchema.get("required", [])
        
        # Build fields dict for Pydantic model
        fields = {}
        for prop_name, prop_info in schema_props.items():
            field_type = str  # Default to string
            
            # Map JSON schema types to Python types
            if prop_info.get("type") == "number":
                field_type = float
            elif prop_info.get("type") == "integer":
                field_type = int
            elif prop_info.get("type") == "boolean":
                field_type = bool
            
            # Create field
            is_required = prop_name in required_fields
            default = ... if is_required else None
            
            fields[prop_name] = (
                field_type,
                Field(
                    default=default,
                    description=prop_info.get("description", "")
                )
            )
        
        # Create Pydantic model dynamically
        InputModel = type(
            f"{mcp_tool.name}Input",
            (BaseModel,),
            {
                "__annotations__": {k: v[0] for k, v in fields.items()},
                **{k: v[1] for k, v in fields.items()}
            }
        )
        
        # Create async function for tool
        async def tool_func(**kwargs):
            """Execute MCP tool"""
            try:
                result = await self.session.call_tool(mcp_tool.name, kwargs)
                
                if result.content:
                    texts = [c.text for c in result.content if hasattr(c, 'text')]
                    return "\n".join(texts) if texts else "No result"
                
                return "No result"
            
            except Exception as e:
                logger.error(f"MCP tool execution error: {e}")
                return f"Error: {str(e)}"
        
        # Create StructuredTool
        return StructuredTool(
            name=mcp_tool.name,
            description=mcp_tool.description,
            args_schema=InputModel,
            coroutine=tool_func
        )
    
    async def cleanup(self):
        """Cleanup MCP connection"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if hasattr(self, 'stdio_context'):
            await self.stdio_context.__aexit__(None, None, None)
        
        logger.info("MCP connection closed")

# ==================== PRODUCTION AGENT ====================

class ProductionMCPState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]

async def create_production_mcp_agent(mcp_server_script: str):
    """Create production-ready agent with MCP tools"""
    
    # Initialize MCP
    mcp_wrapper = MCPToolWrapper(mcp_server_script)
    await mcp_wrapper.initialize()
    
    # Get tools
    tools = mcp_wrapper.langchain_tools
    
    # Initialize LLM with tools
    llm = ChatOllama(model="llama3.2", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # Agent node
    def agent(state: ProductionMCPState):
        """Agent that can call MCP tools"""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    # Tool node (handles async tools automatically)
    tool_node = ToolNode(tools)
    
    # Router
    def should_continue(state: ProductionMCPState):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "end"
    
    # Build graph
    workflow = StateGraph(ProductionMCPState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")
    
    # Compile
    agent_app = workflow.compile()
    
    return agent_app, mcp_wrapper

# Usage
async def run_production_mcp_agent():
    """Run production MCP agent"""
    
    mcp_server_path = "./mcp_server.py"
    
    agent, mcp_wrapper = await create_production_mcp_agent(mcp_server_path)
    
    try:
        queries = [
            "Search the database for laptops",
            "Calculate the final price of a $1299 item with 20% discount",
            "Check availability of product p3 in TX warehouse"
        ]
        
        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}\n")
            
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=query)]
            })
            
            # Print result
            final_message = result["messages"][-1]
            print(f"Agent: {final_message.content}\n")
    
    finally:
        await mcp_wrapper.cleanup()

if __name__ == "__main__":
    asyncio.run(run_production_mcp_agent())
```

---

## 🔧 Part 6: Building Custom MCP Server

### Real-World Example: File System MCP Server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource
from pathlib import Path
import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create server
app = Server("filesystem-mcp-server")

# Safe directory for file operations
SAFE_DIR = Path("./mcp_files")
SAFE_DIR.mkdir(exist_ok=True)

# ==================== TOOLS ====================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List file system tools"""
    return [
        Tool(
            name="list_files",
            description="List files in a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (relative to safe directory)",
                        "default": "."
                    }
                }
            }
        ),
        Tool(
            name="read_file",
            description="Read contents of a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to file"
                    }
                },
                "required": ["filepath"]
            }
        ),
        Tool(
            name="write_file",
            description="Write content to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    }
                },
                "required": ["filepath", "content"]
            }
        ),
        Tool(
            name="search_files",
            description="Search for files containing text",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for"
                    },
                    "file_extension": {
                        "type": "string",
                        "description": "Filter by file extension (e.g., .txt, .py)",
                        "default": ""
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute file system tools"""
    logger.info(f"Tool called: {name}")
    
    try:
        if name == "list_files":
            result = await list_files_impl(arguments.get("path", "."))
        
        elif name == "read_file":
            result = await read_file_impl(arguments.get("filepath", ""))
        
        elif name == "write_file":
            result = await write_file_impl(
                arguments.get("filepath", ""),
                arguments.get("content", "")
            )
        
        elif name == "search_files":
            result = await search_files_impl(
                arguments.get("query", ""),
                arguments.get("file_extension", "")
            )
        
        else:
            result = f"Unknown tool: {name}"
        
        return [TextContent(type="text", text=result)]
    
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

# Tool implementations
async def list_files_impl(path: str) -> str:
    """List files in directory"""
    target_dir = (SAFE_DIR / path).resolve()
    
    # Security check
    if not str(target_dir).startswith(str(SAFE_DIR.resolve())):
        return "Error: Access denied"
    
    if not target_dir.exists():
        return f"Error: Directory '{path}' not found"
    
    if not target_dir.is_dir():
        return f"Error: '{path}' is not a directory"
    
    files = []
    dirs = []
    
    for item in target_dir.iterdir():
        if item.is_file():
            files.append({
                "name": item.name,
                "size": item.stat().st_size,
                "type": "file"
            })
        elif item.is_dir():
            dirs.append({
                "name": item.name,
                "type": "directory"
            })
    
    result = {
        "path": path,
        "directories": dirs,
        "files": files
    }
    
    return json.dumps(result, indent=2)

async def read_file_impl(filepath: str) -> str:
    """Read file content"""
    file_path = (SAFE_DIR / filepath).resolve()
    
    # Security check
    if not str(file_path).startswith(str(SAFE_DIR.resolve())):
        return "Error: Access denied"
    
    if not file_path.exists():
        return f"Error: File '{filepath}' not found"
    
    if not file_path.is_file():
        return f"Error: '{filepath}' is not a file"
    
    # Size limit
    max_size = 1024 * 1024  # 1MB
    if file_path.stat().st_size > max_size:
        return "Error: File too large (max 1MB)"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    
    except UnicodeDecodeError:
        return "Error: File is not text (binary file?)"

async def write_file_impl(filepath: str, content: str) -> str:
    """Write content to file"""
    file_path = (SAFE_DIR / filepath).resolve()
    
    # Security check
    if not str(file_path).startswith(str(SAFE_DIR.resolve())):
        return "Error: Access denied"
    
    # Create parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote {len(content)} characters to '{filepath}'"
    
    except Exception as e:
        return f"Error writing file: {str(e)}"

async def search_files_impl(query: str, file_extension: str) -> str:
    """Search files for text"""
    matches = []
    
    # Search pattern
    pattern = f"*{file_extension}" if file_extension else "*.*"
    
    for file_path in SAFE_DIR.rglob(pattern):
        if not file_path.is_file():
            continue
        
        # Skip large files
        if file_path.stat().st_size > 1024 * 1024:
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if query.lower() in content.lower():
                # Get relative path
                rel_path = file_path.relative_to(SAFE_DIR)
                
                # Find line numbers
                lines = content.split('\n')
                line_numbers = [
                    i + 1 for i, line in enumerate(lines)
                    if query.lower() in line.lower()
                ]
                
                matches.append({
                    "file": str(rel_path),
                    "line_numbers": line_numbers[:5]  # First 5 matches
                })
        
        except:
            continue
    
    result = {
        "query": query,
        "matches_found": len(matches),
        "files": matches[:10]  # First 10 files
    }
    
    return json.dumps(result, indent=2)

# ==================== RESOURCES ====================

@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources"""
    return [
        Resource(
            uri="fs://files",
            name="File System",
            description="Access to all files in the safe directory",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource"""
    if uri == "fs://files":
        # Return directory tree
        result = await list_files_impl(".")
        return result
    
    return f"Unknown resource: {uri}"

# Run server
async def main():
    logger.info("Starting File System MCP Server...")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

**Save as `mcp_filesystem_server.py`**

---

## 📊 Part 7: MCP vs Traditional Tools Comparison

| Aspect | Traditional Tools | MCP Tools |
|--------|-------------------|-----------|
| **Definition** | Ad-hoc function definitions | Standardized protocol |
| **Discovery** | Hardcoded in agent | Dynamic discovery via protocol |
| **Sharing** | Copy-paste code | Share MCP server |
| **Versioning** | Manual | Built into protocol |
| **Multiple Agents** | Redefine for each | One server, many clients |
| **Language Support** | Same language as agent | Any language (protocol-based) |
| **Deployment** | Embedded in agent | Separate server process |
| **Resources** | Not standardized | First-class concept |
| **Prompts** | Agent-defined | Server can provide templates |

---

## 🏆 Part 8: Production Best Practices

### 1. Error Handling in MCP Servers

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Always wrap in try-except"""
    try:
        # Validate inputs first
        if not arguments:
            return [TextContent(type="text", text="Error: Missing arguments")]
        
        # Execute tool
        result = await execute_tool(name, arguments)
        
        return [TextContent(type="text", text=result)]
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return [TextContent(type="text", text=f"Validation error: {str(e)}")]
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]
```

### 2. Resource Management

```python
# Always use context managers
async with stdio_server() as (read, write):
    await app.run(read, write, app.create_initialization_options())

# Cleanup on shutdown
try:
    await mcp_client.cleanup()
finally:
    logger.info("Cleanup complete")
```

### 3. Security

```python
# Validate file paths
def is_safe_path(base: Path, path: str) -> bool:
    """Check if path is within safe directory"""
    target = (base / path).resolve()
    return str(target).startswith(str(base.resolve()))

# Rate limiting
from collections import defaultdict
import time

call_counts = defaultdict(list)

def rate_limit_check(client_id: str, max_calls: int = 100, period: int = 60) -> bool:
    """Check rate limit"""
    now = time.time()
    call_counts[client_id] = [t for t in call_counts[client_id] if t > now - period]
    
    if len(call_counts[client_id]) >= max_calls:
        return False
    
    call_counts[client_id].append(now)
    return True
```

---

## 🧠 Key Concepts to Remember

1. **MCP standardizes tool interfaces** across applications
2. **MCP servers expose tools, resources, and prompts**
3. **MCP clients connect to servers via stdio/HTTP/WebSocket**
4. **Tools, Resources, Prompts are three core MCP concepts**
5. **Use async/await for MCP operations**
6. **Wrap MCP tools in LangChain StructuredTool for bind_tools**
7. **Always validate inputs and handle errors**
8. **Security is critical** - validate paths, rate limit, sanitize inputs
9. **MCP enables tool sharing** across multiple agents
10. **Production MCP servers should log and monitor**

---

## 🚀 What's Next?

In **Chapter 9**, we'll explore:
- **State Management & Memory** in depth
- Short-term vs long-term memory
- Vector stores for semantic memory
- State pruning and compression
- Memory retrieval strategies
- Production memory patterns

---

## ✅ Chapter 8 Complete!

**You now understand:**
- ✅ What MCP is and why it matters
- ✅ MCP architecture (servers, clients, transport)
- ✅ Building production MCP servers
- ✅ Integrating MCP with LangGraph
- ✅ Converting MCP tools to LangChain tools
- ✅ Using bind_tools with MCP tools
- ✅ Custom MCP server creation (file system example)
- ✅ Security and best practices
- ✅ MCP vs traditional tools comparison

**Ready for Chapter 9?** Just say "Continue to Chapter 9" or ask any questions!