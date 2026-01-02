# Tutorial 2: Tool Use - Connecting to Business Logic

**Duration: 45 minutes**

Learn how to give your AI agent "hands" - the ability to take real actions by calling APIs, querying databases, and interacting with business logic.

## What You'll Learn

- **Function/Tool Calling**: How LLMs interface with external systems
- **Structured Outputs**: Using JSON schemas for reliable tool definitions
- **Routing Logic**: How the AI decides when to talk vs. when to act
- **The Tool Loop**: Executing tools and feeding results back to the LLM

## What You'll Build

- A mock e-commerce backend with products, orders, and shipping
- Tool definitions that the AI can understand and use
- An AI shopping assistant that can:
  - Search products
  - Check inventory
  - Look up order status
  - Calculate shipping costs

## Key Concepts

### 1. Tool Definitions

Tools are defined with JSON Schema that tells the AI:
- **What** the tool does (description)
- **What inputs** it needs (parameters)
- **Which inputs** are required vs optional

```python
{
    "type": "function",
    "function": {
        "name": "get_product_info",
        "description": "Get details about a product",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product ID"
                }
            },
            "required": ["product_id"]
        }
    }
}
```

### 2. The Tool Calling Loop

```
User asks question
       ↓
LLM receives message + available tools
       ↓
LLM decides: respond directly OR call tool(s)
       ↓
If tool call: execute tool, return result to LLM
       ↓
LLM incorporates result into response
       ↓
Return final response to user
```

### 3. Deterministic vs Probabilistic

| Aspect | Direct Response | Tool Response |
|--------|-----------------|---------------|
| **Source** | LLM's training | Your actual data |
| **Accuracy** | May hallucinate | Ground truth |
| **Speed** | Faster | Adds latency |
| **Use Case** | General knowledge | Specific/current data |

## Getting Started

### 1. Run the Setup Script

```bash
chmod +x start.sh
./start.sh
```

### 2. Configure Your API Key

Edit `.env` with your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the Tutorial

```bash
./start.sh
```

## Tutorial Flow

1. **Tool Calling Explanation** - Visual walkthrough of how it works
2. **Example Queries** - Watch the AI decide when to use tools
3. **Interactive Session** - Chat with the e-commerce assistant

## Try These Queries

| Query | Expected Tool(s) |
|-------|------------------|
| "Hello!" | None (direct response) |
| "What's the laptop-pro price?" | `get_product_info` |
| "Show me accessories under $100" | `search_products` |
| "Where's my order ORD-002?" | `check_order_status` |
| "Cost for 2 keyboards with express?" | `calculate_shipping` |

## Code Walkthrough

### Tool Implementation

```python
def get_product_info(product_id: str) -> dict:
    """Your actual business logic lives here."""
    product = database.get(product_id)
    return {"name": product.name, "price": product.price}
```

### Tool Execution Loop

```python
while True:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )

    if response.tool_calls:
        # Execute tools and add results
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append({"role": "tool", "content": result})
    else:
        # Final response - no more tools
        return response.content
```

## Product Manager Insights

### Defining Capabilities

When designing AI products, ask:
- What actions should the AI be able to take?
- What data should it access?
- What should it NOT be able to do?

### Tool Description Quality

Good descriptions = Better AI decisions:

| Bad | Good |
|-----|------|
| "Gets data" | "Get detailed product information including price, stock status, and description by product ID" |
| "Search" | "Search products with optional filters for category and maximum price" |

### Error Handling

Consider:
- What if a tool fails?
- What if the product doesn't exist?
- How should the AI communicate errors?

## Exercises

1. **Add a New Tool**: Create a `create_order` tool that places orders
2. **Improve Descriptions**: Make tool descriptions more detailed
3. **Add Error Handling**: Handle out-of-stock products gracefully
4. **Rate Limiting**: Add a tool call limit per conversation

## Common Issues

| Problem | Solution |
|---------|----------|
| "Tool not being called" | Check your tool description - is it clear? |
| "Wrong tool called" | Make descriptions more specific |
| "JSON parse error" | Validate your tool parameters schema |

## Next Steps

Ready to make your AI's decision-making transparent?

**→ Tutorial 3: Reasoning & Transparency (ReAct Pattern)**

---

## Files in This Tutorial

| File | Purpose |
|------|---------|
| `start.sh` | Setup script |
| `tool_agent.py` | Main tutorial with e-commerce assistant |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
