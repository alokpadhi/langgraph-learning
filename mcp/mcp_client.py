from typing import TypedDict, Annotated, Sequence, Any
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import asyncio
import json
import logging
import aiosqlite

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
    conn = await aiosqlite.connect("mcp_agent.db")
    checkpointer = AsyncSqliteSaver(conn)
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
    mcp_server_path = "mcp_server.py"
    
    queries = [
        "Search for electronics products",
        "Calculate the price of a $1299 laptop with 15% discount",
        "Check if product p1 is available in the CA warehouse"
    ]
    
    for query in queries:
        asyncio.run(chat_with_mcp_agent(query, mcp_server_path))
        print("\n" + "="*60 + "\n")