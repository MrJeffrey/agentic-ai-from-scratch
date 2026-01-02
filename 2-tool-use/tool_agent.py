#!/usr/bin/env python3
"""
Tutorial 2: Tool Use - Connecting AI to Business Logic
=======================================================

This tutorial teaches you how to give AI agents the ability to take actions:
- How LLMs interface with tools/APIs
- Structured outputs (JSON) for reliable actions
- The tool calling pattern
- Routing: when to talk vs. when to act

Learning Objectives:
- Understand the tool/function calling mechanism
- Build an agent that can use multiple tools
- See how AI decides WHEN to use which tool
- Handle tool results and multi-step interactions
"""

import os
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI


class Spinner:
    """A simple spinner to show during API calls."""

    def __init__(self, message="Calling OpenAI API"):
        self.message = message
        self.spinning = False
        self.thread = None

    def _spin(self):
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self.spinning:
            sys.stdout.write(f"\r{chars[i]} {self.message}...")
            sys.stdout.flush()
            i = (i + 1) % len(chars)
            time.sleep(0.1)

    def start(self):
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def stop(self):
        self.spinning = False
        if self.thread:
            self.thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 15) + "\r")
        sys.stdout.flush()

# Load environment variables from parent directory (shared across all tutorials)
load_dotenv(Path(__file__).parent.parent / ".env")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =============================================================================
# LESSON 1: Mock Business APIs (Our "Backend")
# =============================================================================

# Simulated product database
PRODUCT_CATALOG = {
    "laptop-pro": {
        "name": "Laptop Pro 15",
        "price": 1299.99,
        "stock": 15,
        "category": "Electronics",
        "description": "High-performance laptop with 16GB RAM and 512GB SSD",
    },
    "wireless-mouse": {
        "name": "Wireless Ergonomic Mouse",
        "price": 49.99,
        "stock": 150,
        "category": "Accessories",
        "description": "Comfortable wireless mouse with 6-month battery life",
    },
    "usb-hub": {
        "name": "USB-C Hub 7-in-1",
        "price": 79.99,
        "stock": 0,  # Out of stock!
        "category": "Accessories",
        "description": "Expand your ports with HDMI, USB-A, and SD card slots",
    },
    "monitor-4k": {
        "name": "UltraWide 4K Monitor",
        "price": 599.99,
        "stock": 8,
        "category": "Electronics",
        "description": "34-inch curved 4K display with HDR support",
    },
    "keyboard-mech": {
        "name": "Mechanical Keyboard RGB",
        "price": 129.99,
        "stock": 42,
        "category": "Accessories",
        "description": "Cherry MX switches with customizable RGB lighting",
    },
}

# Simulated order history
ORDER_HISTORY = {
    "ORD-001": {
        "product": "laptop-pro",
        "quantity": 1,
        "total": 1299.99,
        "status": "delivered",
        "date": "2024-01-15",
    },
    "ORD-002": {
        "product": "wireless-mouse",
        "quantity": 2,
        "total": 99.98,
        "status": "shipped",
        "date": "2024-01-20",
    },
    "ORD-003": {
        "product": "keyboard-mech",
        "quantity": 1,
        "total": 129.99,
        "status": "processing",
        "date": "2024-01-22",
    },
}


# =============================================================================
# LESSON 2: Tool Implementations (What tools actually DO)
# =============================================================================

def get_product_info(product_id: str) -> dict:
    """
    Get information about a specific product.

    This simulates calling your product database API.
    """
    product = PRODUCT_CATALOG.get(product_id.lower())
    if product:
        return {
            "success": True,
            "product": {
                "id": product_id,
                **product,
                "in_stock": product["stock"] > 0,
            },
        }
    return {
        "success": False,
        "error": f"Product '{product_id}' not found",
        "available_products": list(PRODUCT_CATALOG.keys()),
    }


def search_products(query: str = None, category: str = None, max_price: float = None) -> dict:
    """
    Search products with optional filters.

    This simulates a search API endpoint.
    """
    results = []

    for pid, product in PRODUCT_CATALOG.items():
        # Apply filters
        if query and query.lower() not in product["name"].lower():
            continue
        if category and product["category"].lower() != category.lower():
            continue
        if max_price and product["price"] > max_price:
            continue

        results.append({
            "id": pid,
            "name": product["name"],
            "price": product["price"],
            "in_stock": product["stock"] > 0,
        })

    return {
        "success": True,
        "count": len(results),
        "products": results,
    }


def check_order_status(order_id: str) -> dict:
    """
    Check the status of an order.

    This simulates an order tracking API.
    """
    order = ORDER_HISTORY.get(order_id.upper())
    if order:
        product = PRODUCT_CATALOG.get(order["product"], {})
        return {
            "success": True,
            "order": {
                "order_id": order_id.upper(),
                "product_name": product.get("name", "Unknown"),
                "quantity": order["quantity"],
                "total": order["total"],
                "status": order["status"],
                "order_date": order["date"],
            },
        }
    return {
        "success": False,
        "error": f"Order '{order_id}' not found",
        "hint": "Order IDs look like: ORD-001, ORD-002, etc.",
    }


def calculate_shipping(product_id: str, quantity: int, express: bool = False) -> dict:
    """
    Calculate shipping cost for an order.

    This simulates a shipping calculator API.
    """
    product = PRODUCT_CATALOG.get(product_id.lower())
    if not product:
        return {"success": False, "error": f"Product '{product_id}' not found"}

    base_cost = 5.99
    per_item = 2.00
    express_multiplier = 2.5 if express else 1.0

    shipping_cost = (base_cost + (per_item * quantity)) * express_multiplier
    subtotal = product["price"] * quantity
    total = subtotal + shipping_cost

    return {
        "success": True,
        "calculation": {
            "product": product["name"],
            "quantity": quantity,
            "unit_price": product["price"],
            "subtotal": round(subtotal, 2),
            "shipping_type": "Express" if express else "Standard",
            "shipping_cost": round(shipping_cost, 2),
            "total": round(total, 2),
            "estimated_delivery": "1-2 days" if express else "5-7 days",
        },
    }


def get_current_time() -> dict:
    """
    Get the current date and time.

    Simple utility tool to show tools can do anything.
    """
    now = datetime.now()
    return {
        "success": True,
        "datetime": now.isoformat(),
        "formatted": now.strftime("%B %d, %Y at %I:%M %p"),
    }


# =============================================================================
# LESSON 3: Tool Definitions (What the AI "sees")
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Get detailed information about a specific product by its ID. Use this when the user asks about a specific product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID (e.g., 'laptop-pro', 'wireless-mouse')",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products with optional filters. Use this when the user wants to browse or find products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to find in product names",
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (Electronics or Accessories)",
                        "enum": ["Electronics", "Accessories"],
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price filter",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check the status of an existing order. Use this when the user asks about their order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID (e.g., 'ORD-001')",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_shipping",
            "description": "Calculate shipping cost and total for a potential order. Use this when the user wants to know the total cost including shipping.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID to calculate shipping for",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of items to order",
                        "minimum": 1,
                    },
                    "express": {
                        "type": "boolean",
                        "description": "Whether to use express shipping",
                        "default": False,
                    },
                },
                "required": ["product_id", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time. Use when the user asks what time it is.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Map function names to implementations
TOOL_FUNCTIONS = {
    "get_product_info": get_product_info,
    "search_products": search_products,
    "check_order_status": check_order_status,
    "calculate_shipping": calculate_shipping,
    "get_current_time": get_current_time,
}


# =============================================================================
# LESSON 4: The Tool-Using Agent
# =============================================================================

SYSTEM_PROMPT = """You are a helpful e-commerce assistant for TechShop, an online electronics store.

You have access to tools that let you:
- Search for products and get product details
- Check order status for customers
- Calculate shipping costs

Guidelines:
1. Always use tools to get accurate, real-time information
2. If a product is out of stock, let the customer know and suggest alternatives
3. Be helpful and conversational while being accurate
4. If you're not sure about something, use the appropriate tool to find out
5. IMPORTANT: Do NOT guess product prices, stock levels, or order statuses.
   If you don't have the information, you MUST use a tool to retrieve it.

Available products include laptops, monitors, keyboards, mice, and USB accessories.
Order IDs are in the format ORD-XXX (e.g., ORD-001, ORD-002).
"""


def execute_tool(tool_name: str, tool_args: dict) -> Any:
    """Execute a tool and return its result, catching any errors gracefully."""
    try:
        if tool_name in TOOL_FUNCTIONS:
            return TOOL_FUNCTIONS[tool_name](**tool_args)
        return {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        # Return error to LLM so it can apologize to the user
        return {"error": f"Tool execution failed: {str(e)}"}


def run_agent(user_message: str, messages: list, verbose: bool = True, max_turns: int = 8) -> tuple[str, list]:
    """
    Run the tool-using agent for one turn.

    This implements the TOOL CALLING LOOP:
    1. Send messages to LLM
    2. If LLM wants to call tools, execute them
    3. Send results back to LLM
    4. Repeat until LLM gives final response
    """

    # Add user message
    messages.append({"role": "user", "content": user_message})

    turn_count = 0

    while True:
        turn_count += 1
        if turn_count > max_turns:
            safety_msg = "Stopping to avoid an infinite tool loop. Please rephrase or simplify your request."
            messages.append({"role": "assistant", "content": safety_msg})
            return safety_msg, messages

        # Call the LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # Let the model decide
        )

        response_message = response.choices[0].message

        # Check if we need to call tools
        if response_message.tool_calls:
            # Add the assistant's response (with tool calls) to history
            # Using model_dump() ensures we store a clean dictionary, not an OpenAI object.
            # This prevents issues when serializing the conversation to JSON later.
            messages.append(response_message.model_dump())

            if verbose:
                print(f"\n  🔧 Agent is calling {len(response_message.tool_calls)} tool(s)...")

            # IMPORTANT: The LLM does NOT execute code!
            # It pauses generation and tells us which function it wants to run.
            # WE (the Python script) execute the tool and return the result.

            # NOTE: The loop below handles "parallel function calling" -
            # when the user asks two things at once (e.g., "check order A and B"),
            # the LLM can request multiple tool calls in a single turn.
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name

                # Parse arguments, handling cases where LLM generates invalid JSON
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                if verbose:
                    print(f"     → {tool_name}({json.dumps(tool_args)})")

                # Execute the tool
                result = execute_tool(tool_name, tool_args)

                if verbose:
                    print(f"     ← {json.dumps(result, indent=2)[:200]}...")

                # Add tool result to messages
                # We include tool_call_id so the LLM knows which request this answers
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result),
                })

        else:
            # No tool calls - we have a final response
            # Using model_dump() for consistency (clean dict, not OpenAI object)
            messages.append(response_message.model_dump())
            return response_message.content, messages


# =============================================================================
# Interactive Demo
# =============================================================================

def demonstrate_tool_calling():
    """Show how tool calling works with examples."""

    print("\n" + "=" * 60)
    print("LESSON: How Tool Calling Works")
    print("=" * 60)

    print("""
When an LLM has tools available, it can choose to:
1. Respond directly (no tools needed)
2. Call one or more tools
3. Use tool results to form a response

The FLOW looks like this:

┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  User   │ ──► │   LLM   │ ──► │  Tool   │ ──► │   LLM   │
│ Message │     │ Decides │     │ Executes│     │ Responds│
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                     │                              ▲
                     │    (tool result)             │
                     └──────────────────────────────┘

KEY INSIGHT: The LLM decides WHEN to use tools based on:
- The user's intent
- The tool descriptions you provide
- Whether it has the information already
""")

    input("\nPress Enter to see some examples...")

    # Example conversations
    examples = [
        {
            "title": "Example 1: Direct Response (No Tool Needed)",
            "query": "Hello! How are you?",
            "explanation": "Greetings don't need tools - the AI responds directly.",
        },
        {
            "title": "Example 2: Single Tool Call",
            "query": "What's the price of the laptop-pro?",
            "explanation": "The AI calls get_product_info to get accurate data.",
        },
        {
            "title": "Example 3: Search and Filter",
            "query": "Show me accessories under $100",
            "explanation": "The AI uses search_products with filters.",
        },
        {
            "title": "Example 4: Multi-Step Query",
            "query": "How much would it cost to order 2 keyboards with express shipping?",
            "explanation": "The AI might call search first, then calculate_shipping.",
        },
    ]

    for example in examples:
        print(f"\n{'─' * 50}")
        print(f"  {example['title']}")
        print(f"{'─' * 50}")
        print(f"  Query: \"{example['query']}\"")
        print(f"  Why: {example['explanation']}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        response, _ = run_agent(example["query"], messages, verbose=True)

        print(f"\n  🤖 Response: {response[:300]}{'...' if len(response) > 300 else ''}")

        input("\n  Press Enter for next example...")


def run_interactive_session():
    """Run an interactive session with the tool-using agent."""

    print("\n" + "=" * 60)
    print("INTERACTIVE SESSION: E-Commerce Assistant")
    print("=" * 60)

    print("""
Now you can chat with the tool-using agent!

Try asking things like:
  • "What products do you have?"
  • "Tell me about the monitor"
  • "What's the status of order ORD-002?"
  • "How much for 3 mice with express shipping?"
  • "Show me electronics under $1000"

Type 'quit' to exit, 'tools' to see available tools.
""")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("\n👤 You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break

        if user_input.lower() == 'tools':
            print("\n📋 Available Tools:")
            for tool in TOOLS:
                func = tool["function"]
                print(f"   • {func['name']}: {func['description'][:60]}...")
            continue

        spinner = Spinner("Calling OpenAI API")
        spinner.start()
        response, messages = run_agent(user_input, messages, verbose=False)
        spinner.stop()
        print(f"\n🤖 Assistant: {response}")

        # Show inline stats and command hints
        msg_count = len([m for m in messages if isinstance(m, dict) and m.get("role") == "user"])
        # Count tool calls from assistant messages (now stored as dicts via model_dump)
        tool_call_count = len([m for m in messages if isinstance(m, dict) and m.get("tool_calls")])
        print(f"\n  [{msg_count} messages · {tool_call_count} tool calls] 'tools' list | 'quit' exit")

    print("\nSession ended. Tools called during this session:")
    tool_call_turns = [m for m in messages if isinstance(m, dict) and m.get("tool_calls")]
    print(f"   Total tool-calling turns: {len(tool_call_turns)}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the complete tutorial."""

    print("""
╔════════════════════════════════════════════════════════════╗
║     Tutorial 2: Tool Use                                   ║
║     Connecting AI to Business Logic                        ║
╚════════════════════════════════════════════════════════════╝

This tutorial covers:
  • Defining tools with JSON schemas
  • The tool calling loop
  • Routing: when the AI decides to use tools
  • Building an e-commerce assistant

Let's see how AI agents take ACTION!
""")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("Please create a .env file with your API key.")
        return

    input("Press Enter to start the demo...")

    demonstrate_tool_calling()
    run_interactive_session()

    print("""
╔════════════════════════════════════════════════════════════╗
║     Tutorial Complete!                                      ║
╚════════════════════════════════════════════════════════════╝

What you learned:
  ✓ Tools let AI take actions (not just talk)
  ✓ Tool definitions tell the AI what's available
  ✓ The AI decides when to use tools based on context
  ✓ Tool results feed back into the conversation

Product Manager Insights:
  • Tools = Your product's capabilities
  • Good tool descriptions = Better AI decisions
  • Consider: What actions should your AI be able to take?

Pro Tip for Developers:
  Writing JSON schemas manually (like the TOOLS list) is error-prone.
  In production, use libraries like 'pydantic' or 'instructor' to
  auto-generate tool schemas from Python function signatures.

Next Steps:
  → Tutorial 3: ReAct Pattern - Make your AI "think" before acting
""")


if __name__ == "__main__":
    main()
