#!/usr/bin/env python3
"""
Tutorial 3: Reasoning & Transparency (ReAct Pattern)
=====================================================

This tutorial teaches you how to make AI decision-making transparent:
- The ReAct Pattern: Reasoning + Acting
- Chain of Thought (CoT) prompting
- Creating auditable AI systems
- Trade-offs between speed and accuracy

Learning Objectives:
- Understand why "thinking before acting" improves accuracy
- Build an agent that shows its reasoning
- Learn to debug AI decisions through reasoning traces
- Balance latency vs. accuracy in production
"""

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from parent directory (shared across all tutorials)
load_dotenv(Path(__file__).parent.parent / ".env")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


# =============================================================================
# LESSON 1: Why Reasoning Matters
# =============================================================================

def demonstrate_reasoning_value():
    """Show how explicit reasoning improves accuracy."""

    print("\n" + "=" * 60)
    print("LESSON 1: Why Reasoning Matters")
    print("=" * 60)

    print("""
The ReAct Pattern forces the AI to:
1. REASON about the problem first
2. Then ACT based on that reasoning

Why does this help?
┌─────────────────────────────────────────────────────────────┐
│  WITHOUT Reasoning:                                         │
│  User asks → AI immediately answers → May be wrong          │
│                                                             │
│  WITH Reasoning:                                            │
│  User asks → AI thinks step by step → More accurate answer  │
└─────────────────────────────────────────────────────────────┘

Benefits:
  • Fewer errors on complex problems
  • Auditable decision-making (you can see WHY)
  • Easier debugging when things go wrong
  • Better for compliance/regulated industries

Trade-off: More tokens = more latency and cost
""")

    # Demonstrate with a tricky math problem
    tricky_problem = "If I have 3 apples and give away half, then buy 5 more, how many do I have?"

    print("Let's compare direct answering vs. reasoning...\n")
    print(f"Problem: {tricky_problem}\n")

    # Direct answering
    print("─" * 50)
    print("APPROACH 1: Direct Answer (no reasoning)")
    print("─" * 50)

    direct_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer questions concisely in one sentence."},
            {"role": "user", "content": tricky_problem},
        ],
        max_tokens=50,
    )
    print(f"Response: {direct_response.choices[0].message.content}")
    print(f"Tokens used: {direct_response.usage.total_tokens}")

    # With reasoning
    print("\n" + "─" * 50)
    print("APPROACH 2: Chain of Thought (explicit reasoning)")
    print("─" * 50)

    cot_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Think through problems step by step. Show your reasoning, then give the answer. Use plain text only - no LaTeX or mathematical notation like \\frac or \\[.",
            },
            {"role": "user", "content": tricky_problem},
        ],
        max_tokens=200,
    )
    print(f"Response: {cot_response.choices[0].message.content}")
    print(f"Tokens used: {cot_response.usage.total_tokens}")

    print("\n💡 Notice: Reasoning uses more tokens but is more reliable!\n")

    input("Press Enter to continue to the ReAct Pattern...")


# =============================================================================
# LESSON 2: The ReAct Pattern
# =============================================================================

# Knowledge base for the agent to query
KNOWLEDGE_BASE = {
    "company_policy": {
        "refund_window": 30,
        "refund_requirements": "Original receipt required, item must be unused",
        "exceptions": "Electronics have 15-day window, final sale items non-refundable",
    },
    "shipping_info": {
        "standard": "5-7 business days, free over $50",
        "express": "2-3 business days, $9.99",
        "overnight": "Next business day, $24.99",
    },
    "store_hours": {
        "weekdays": "9 AM - 9 PM",
        "saturday": "10 AM - 8 PM",
        "sunday": "11 AM - 6 PM",
        "holidays": "Hours may vary, check website",
    },
    "loyalty_program": {
        "points_per_dollar": 1,
        "redemption_rate": "100 points = $1 off",
        "elite_threshold": 500,
        "elite_benefits": "Free express shipping, early access to sales",
    },
}


def lookup_policy(topic: str) -> dict:
    """Look up information from the knowledge base."""
    topic_lower = topic.lower().replace(" ", "_")
    if topic_lower in KNOWLEDGE_BASE:
        return {"found": True, "data": KNOWLEDGE_BASE[topic_lower]}
    return {"found": False, "available_topics": list(KNOWLEDGE_BASE.keys())}


def calculate_points(purchase_amount: float, is_elite: bool = False) -> dict:
    """Calculate loyalty points for a purchase."""
    base_points = int(purchase_amount)
    multiplier = 2 if is_elite else 1
    total_points = base_points * multiplier
    return {
        "purchase_amount": purchase_amount,
        "is_elite": is_elite,
        "base_points": base_points,
        "multiplier": multiplier,
        "total_points": total_points,
    }


def check_refund_eligibility(days_since_purchase: int, item_type: str, has_receipt: bool) -> dict:
    """Check if a refund is eligible."""
    # Electronics have 15-day window
    window = 15 if item_type.lower() == "electronics" else 30

    eligible = days_since_purchase <= window and has_receipt

    return {
        "days_since_purchase": days_since_purchase,
        "item_type": item_type,
        "refund_window": window,
        "has_receipt": has_receipt,
        "eligible": eligible,
        "reason": "Eligible for refund" if eligible else f"Outside {window}-day window or missing receipt",
    }


REACT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": "Look up company policies and information. Available topics: 'company_policy' (for refunds, returns, exchanges), 'shipping_info' (delivery options and costs), 'store_hours' (operating hours), 'loyalty_program' (points and rewards). You must use one of these exact topic names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": ["company_policy", "shipping_info", "store_hours", "loyalty_program"],
                        "description": "The policy topic to look up",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_points",
            "description": "Calculate loyalty points earned for a purchase amount",
            "parameters": {
                "type": "object",
                "properties": {
                    "purchase_amount": {
                        "type": "number",
                        "description": "The purchase amount in dollars",
                    },
                    "is_elite": {
                        "type": "boolean",
                        "description": "Whether the customer is elite status",
                        "default": False,
                    },
                },
                "required": ["purchase_amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_refund_eligibility",
            "description": "Check if a purchase is eligible for refund",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_since_purchase": {
                        "type": "integer",
                        "description": "Number of days since the purchase",
                    },
                    "item_type": {
                        "type": "string",
                        "description": "Type of item (e.g., 'electronics', 'clothing')",
                    },
                    "has_receipt": {
                        "type": "boolean",
                        "description": "Whether the customer has the receipt",
                    },
                },
                "required": ["days_since_purchase", "item_type", "has_receipt"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "lookup_policy": lookup_policy,
    "calculate_points": calculate_points,
    "check_refund_eligibility": check_refund_eligibility,
}


# =============================================================================
# LESSON 3: Building a ReAct Agent
# =============================================================================

REACT_SYSTEM_PROMPT = """You are a helpful customer service agent for a retail store.

IMPORTANT: You must think step-by-step before taking any action. For each request:

1. THOUGHT: First, analyze what the customer is asking and what information you need
2. ACTION: Use the appropriate tool(s) to get accurate information
3. OBSERVATION: Review what the tool returned
4. THOUGHT: Analyze the results and determine if you have enough information
5. ANSWER: Provide a clear, helpful response to the customer

CRITICAL RULES:
- For ANY question about refunds, returns, or exchanges, use lookup_policy with topic "company_policy"
- For ANY question about shipping, use lookup_policy with topic "shipping_info"
- NEVER guess or make assumptions about policies - only state facts from tool results
- If a tool returns information, use ONLY that information in your answer
- Use check_refund_eligibility to determine if a specific return is allowed

Be helpful, accurate, and transparent about your reasoning.
"""


def run_react_agent(user_message: str, verbose: bool = True) -> str:
    """
    Run the ReAct agent with explicit reasoning.

    The agent will show its thought process before acting.
    """
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if verbose:
        print(f"\n👤 Customer: {user_message}")
        print("─" * 50)

    iteration = 0
    max_iterations = 5

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=REACT_TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # If there are tool calls, execute them
        if message.tool_calls:
            messages.append(message)

            # Print the thought/reasoning if present (even with tool calls)
            # This captures the "THOUGHT:" portion when the model also triggers a tool
            if verbose and message.content:
                print(f"\n💭 {message.content}")

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(f"\n🔧 Using tool: {func_name}")
                    print(f"   Args: {func_args}")

                # Execute the tool
                result = TOOL_FUNCTIONS[func_name](**func_args)

                if verbose:
                    print(f"   Result: {json.dumps(result, indent=2)}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })
        else:
            # No more tool calls - return the response
            if verbose:
                print(f"\n{message.content}")

            return message.content

    return "I apologize, but I couldn't complete your request. Please try again."


def demonstrate_react_pattern():
    """Show the ReAct pattern in action with complex queries."""

    examples = [
        {
            "title": "Simple Query",
            "description": "A straightforward lookup that requires one tool call.",
            "query": "What are your store hours on Saturday?",
        },
        {
            "title": "Multi-Step Reasoning",
            "description": "A query requiring the agent to check multiple conditions.",
            "query": "I bought a laptop 20 days ago and want to return it. I have the receipt. Can I get a refund?",
        },
        {
            "title": "Complex Calculation",
            "description": "Combining policy lookup with calculations.",
            "query": "If I'm an elite member and spend $150, how many points will I earn? And how much is that worth in dollars?",
        },
    ]

    for i, example in enumerate(examples):
        if i == 0:
            clear_screen()

        print("=" * 60)
        print("LESSON 2: The ReAct Pattern in Action")
        print("=" * 60)
        print("""
ReAct = Reasoning + Acting

The pattern: THOUGHT -> ACTION -> OBSERVATION -> THOUGHT -> ANSWER

Watch how the agent thinks before acting...
""")
        print("-" * 60)
        print(f"Example {i + 1} of {len(examples)}: {example['title']}")
        print("-" * 60)
        print(f"\n{example['description']}\n")

        run_react_agent(example["query"])

        if i < len(examples) - 1:
            while input("\n\nType 'next' to continue: ").strip().lower() not in ('next', 'n'):
                print("Type 'next' or 'n' to continue.")
            clear_screen()
        else:
            while input("\n\nType 'next' to continue to Lesson 3: ").strip().lower() not in ('next', 'n'):
                print("Type 'next' or 'n' to continue.")
            clear_screen()


# =============================================================================
# LESSON 4: Comparing Approaches
# =============================================================================

def compare_approaches():
    """Compare direct vs. ReAct approaches."""

    print("\n" + "=" * 60)
    print("LESSON 3: Direct vs. ReAct Comparison")
    print("=" * 60)

    test_query = "I bought headphones (not electronics category) 25 days ago but lost my receipt. Can I return them?"

    print(f"\nTest Query: {test_query}\n")

    # Direct approach
    print("─" * 50)
    print("APPROACH 1: Direct Response")
    print("─" * 50)

    direct_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful retail customer service agent. Answer questions directly."},
            {"role": "user", "content": test_query},
        ],
        max_tokens=150,
    )
    print(f"Response: {direct_response.choices[0].message.content}")
    print(f"Tokens: {direct_response.usage.total_tokens}")

    input("\n\nPress Enter to see Approach 2...")

    # ReAct approach
    print("\n" + "─" * 50)
    print("APPROACH 2: ReAct (with reasoning and tools)")
    print("─" * 50)

    # Run ReAct with verbose output
    run_react_agent(test_query)

    print("""

COMPARISON:
┌────────────────┬──────────────────────────────────────────────────┐
│ Direct         │ Fast but may miss policy details                 │
│                │ Answer based on general knowledge                │
│                │ No audit trail of reasoning                      │
├────────────────┼──────────────────────────────────────────────────┤
│ ReAct          │ Slower but more accurate                         │
│                │ Uses actual policy data                          │
│                │ Full reasoning trace for debugging               │
│                │ Auditable for compliance                         │
└────────────────┴──────────────────────────────────────────────────┘
""")


# =============================================================================
# LESSON 5: Interactive Session
# =============================================================================

def run_interactive_session():
    """Run an interactive session with the ReAct agent."""

    print("\n" + "=" * 60)
    print("INTERACTIVE SESSION: ReAct Customer Service Agent")
    print("=" * 60)

    print("""
Now you can interact with the reasoning agent!

Try complex questions like:
  • "Can I return a TV I bought 10 days ago with receipt?"
  • "How do I become an elite member?"
  • "If I spend $200 as a regular member, how much will my points be worth?"
  • "What's the difference between express and overnight shipping?"

Type 'quit' to exit.
""")

    query_count = 0

    while True:
        user_input = input("\n👤 You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break

        query_count += 1
        # verbose=True shows ReAct reasoning (Thought → Action → Observation)
        # This is the pedagogical point - users see the agent's thinking process
        response = run_react_agent(user_input, verbose=True)
        print(f"\n🤖 Agent: {response}")

        # Show inline stats and command hints
        print(f"\n  [{query_count} queries] 'quit' exit")

    print("\n📊 Session Summary:")
    print(f"   Queries handled: {query_count}")
    print("   The ReAct agent showed its reasoning for each response.")
    print("   In production, you can log these traces for debugging and compliance.")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the complete tutorial."""

    print("""
╔════════════════════════════════════════════════════════════╗
║     Tutorial 3: Reasoning & Transparency                   ║
║     The ReAct Pattern for Auditable AI                     ║
╚════════════════════════════════════════════════════════════╝

This tutorial covers:
  • Why explicit reasoning improves accuracy
  • The ReAct pattern (Reasoning + Acting)
  • Creating auditable AI decisions
  • Trade-offs between speed and accuracy

Let's make AI thinking visible!
""")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("Please create a .env file with your API key.")
        return

    input("Press Enter to start...")

    demonstrate_reasoning_value()
    demonstrate_react_pattern()
    compare_approaches()
    run_interactive_session()

    print("""
╔════════════════════════════════════════════════════════════╗
║     Tutorial Complete!                                      ║
╚════════════════════════════════════════════════════════════╝

What you learned:
  ✓ Chain of Thought improves accuracy on complex tasks
  ✓ ReAct = Reasoning + Acting in a loop
  ✓ Reasoning traces enable debugging and auditing
  ✓ Trade-off: More tokens for better accuracy

Product Manager Insights:
  • Use ReAct for high-stakes decisions (compliance, safety)
  • Direct answers for simple queries (latency-sensitive)
  • Reasoning traces are gold for debugging production issues
  • Consider: Should reasoning be visible to users?

Next Steps:
  → Tutorial 4: Memory & Context - Make your AI remember users!
""")


if __name__ == "__main__":
    main()
