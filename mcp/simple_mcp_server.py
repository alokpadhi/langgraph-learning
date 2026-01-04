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