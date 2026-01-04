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