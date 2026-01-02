#!/usr/bin/env python3
"""
Tutorial 1: The Interaction Loop
================================

This tutorial teaches you the fundamentals of building a conversational AI interface:
- The Request/Response lifecycle
- Managing the Context Window (token economy)
- Using System Prompts to define persona and constraints
- Building a functional chat interface

Learning Objectives:
- Understand how messages flow between user and AI
- Learn about token limits and their cost/latency implications
- See how system prompts shape AI behavior
"""

import os
import sys
import threading
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Load environment variables from parent directory (shared across all tutorials)
load_dotenv(Path(__file__).parent.parent / ".env")


class Spinner:
    """A simple spinner to show during API calls."""

    def __init__(self, message="Thinking"):
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
        sys.stdout.write("\r" + " " * (len(self.message) + 15) + "\r")  # Clear line
        sys.stdout.flush()


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================================================================
# LESSON 1: Understanding Tokens
# =============================================================================

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in a text string.

    CONCEPT: Tokens are the units LLMs use to process text.
    - 1 token ≈ 4 characters or ~0.75 words in English
    - Pricing is based on tokens (input + output)
    - Context windows have token limits

    Why this matters for Product Managers:
    - More tokens = higher cost and latency
    - Context window limits how much "memory" the AI has
    - You must design conversations that fit within limits
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def demonstrate_token_economy():
    """Show how tokens work and their implications."""

    print("\n" + "=" * 60)
    print("LESSON 1: Understanding the Token Economy")
    print("=" * 60)

    examples = [
        "Hello",
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "As a Technical Product Manager, I need to understand the trade-offs between latency, cost, and capability when designing AI-powered features.",
    ]

    print("\nTokens are how LLMs 'see' text. Here's how different texts tokenize:\n")

    for text in examples:
        tokens = count_tokens(text)
        chars = len(text)
        print(f"  '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  → {tokens} tokens, {chars} characters, ratio: {chars/tokens:.1f} chars/token\n")

    print("KEY INSIGHT: English averages ~4 characters per token.")
    print("Pricing snapshot (as of 2025-12-23):")
    print("  • gpt-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens")
    print("  • gpt-4o: ~$5.00 per 1M input tokens, ~$15.00 per 1M output tokens")
    print("Always verify current pricing and context limits at https://platform.openai.com/pricing.\n")

    input("Press Enter to continue to Lesson 2...")


# =============================================================================
# LESSON 2: The Message Structure
# =============================================================================

def demonstrate_message_structure():
    """Show the anatomy of a conversation with an LLM."""

    print("\n" + "=" * 60)
    print("LESSON 2: The Message Structure")
    print("=" * 60)

    print("""
Every LLM conversation is a list of messages with ROLES:

┌─────────────────────────────────────────────────────────────┐
│  SYSTEM    │ Defines the AI's persona, rules, and context  │
│            │ This is your "product spec" for the AI        │
├─────────────────────────────────────────────────────────────┤
│  USER      │ The human's input (questions, commands)       │
│            │ This is what your customers type              │
├─────────────────────────────────────────────────────────────┤
│  ASSISTANT │ The AI's responses                            │
│            │ Previous responses become context for next    │
└─────────────────────────────────────────────────────────────┘

The ENTIRE conversation is sent with each request!
This is why context management matters.
""")

    # Show a simple example
    example_messages = [
        {"role": "system", "content": "You are a helpful assistant for a pizza shop."},
        {"role": "user", "content": "What toppings do you have?"},
        {"role": "assistant", "content": "We have pepperoni, mushrooms, olives, and bell peppers!"},
        {"role": "user", "content": "I'll take pepperoni please."},
    ]

    print("Example conversation structure:\n")
    total_tokens = 0
    for msg in example_messages:
        tokens = count_tokens(msg["content"])
        total_tokens += tokens
        print(f"  [{msg['role'].upper():9}] ({tokens:3} tokens): {msg['content'][:50]}...")

    print(f"\n  Total context so far: {total_tokens} tokens")
    print("  Each new message adds to this total!\n")

    input("Press Enter to continue to Lesson 3...")


# =============================================================================
# LESSON 3: System Prompts as Product Specs
# =============================================================================

# Different personas to demonstrate system prompt power
PERSONAS = {
    "professional": {
        "name": "Corporate Assistant",
        "description": "Watch for: formal language, no slang, straight to business",
        "system_prompt": """You are a professional corporate assistant.
Your communication style is:
- Formal and business-appropriate
- Concise and to the point
- Uses proper grammar and punctuation
- Avoids slang or casual language
Always maintain a professional tone.""",
    },
    "friendly": {
        "name": "Friendly Helper",
        "description": "Watch for: warm greetings, encouraging phrases, casual tone",
        "system_prompt": """You are a warm, friendly assistant who loves helping people!
Your communication style is:
- Casual and conversational
- Uses friendly expressions like "Great question!" or "Happy to help!"
- Supportive and encouraging
- Keeps things light and approachable
Be genuine and make the user feel comfortable.""",
    },
    "technical": {
        "name": "Technical Expert",
        "description": "Watch for: precise terminology, structured format, detailed steps",
        "system_prompt": """You are a technical expert assistant.
Your communication style is:
- Precise and technically accurate
- Uses appropriate technical terminology
- Provides detailed explanations when relevant
- Cites specific examples or documentation
- Structured responses with clear organization
Focus on accuracy and depth of information.""",
    },
    "pirate": {
        "name": "Pirate Captain",
        "description": "Watch for: 'arrr', 'matey', nautical references, stays in character",
        "system_prompt": """You are a pirate captain assistant! Arrr!
Your communication style is:
- Uses pirate speak (arrr, matey, ye, etc.)
- References ships, treasure, and the sea
- Enthusiastic and adventurous
- Still helpful, just in a pirate way!
Never break character - you're a genuine pirate helping landlubbers!""",
    },
}


def demonstrate_personas():
    """Show how system prompts dramatically change AI behavior."""

    print("\n" + "=" * 60)
    print("LESSON 3: System Prompts Define Your Product")
    print("=" * 60)

    print("""
The SYSTEM prompt is your product specification for the AI.
It defines:
- Persona (who the AI is)
- Constraints (what it can/cannot discuss)
- Style (how it communicates)
- Rules (specific behaviors to follow)

Let's see the SAME question answered by different personas:
""")

    input("Press Enter to see how different personas respond to the same question...")

    test_question = "How do I make coffee?"

    total_personas = len(PERSONAS)
    for i, (key, persona) in enumerate(PERSONAS.items(), 1):
        print(f"\n{'─' * 60}")
        print(f"PERSONA {i}/{total_personas}: {persona['name']}")
        print(f"{persona['description']}")
        print(f"{'─' * 60}")

        # Show what we're sending to the API
        print(f"\nSYSTEM PROMPT:")
        for line in persona["system_prompt"].strip().split('\n'):
            print(f"  {line}")

        print(f"\nUSER MESSAGE:")
        print(f'  "{test_question}"')

        input("\nPress Enter to send this to the API...")

        print("Calling OpenAI API...", end=" ", flush=True)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": persona["system_prompt"]},
                {"role": "user", "content": test_question},
            ],
            max_tokens=150,
        )
        print("done!")

        print(f"\nRESPONSE:")
        print(f"  {response.choices[0].message.content}\n")

        if i < total_personas:
            input("Press Enter to see the next persona...")
            print("\033[2J\033[H", end="")  # Clear screen

    print("\nKEY INSIGHT: Same model, same question, completely different products!")
    print("Your system prompt IS your product definition.\n")

    input("Press Enter to continue to the Interactive Chat...")


# =============================================================================
# LESSON 4: Building the Interactive Loop
# =============================================================================

class ChatSession:
    """
    A complete chat session with context management.

    This is the core "Interaction Loop" pattern that powers most AI products.
    """

    def __init__(self, system_prompt: str, max_context_tokens: int = 4000):
        """
        Initialize a chat session.

        Args:
            system_prompt: The AI's persona and rules
            max_context_tokens: Maximum tokens to keep in context
        """
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.messages = [{"role": "system", "content": system_prompt}]
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _count_context_tokens(self) -> int:
        """Count total tokens in current context."""
        total = 0
        for msg in self.messages:
            total += count_tokens(msg["content"])
        return total

    def _trim_context(self):
        """
        Trim old messages to stay within token limits.

        CONCEPT: Context Window Management
        - Keep system prompt (always first)
        - Remove oldest messages when approaching limit
        - This is a simple strategy; production systems may summarize
        """
        while self._count_context_tokens() > self.max_context_tokens and len(self.messages) > 2:
            # Remove the oldest non-system message
            self.messages.pop(1)

    def send_message(self, user_input: str) -> str:
        """
        Send a message and get a response.

        This is the core REQUEST/RESPONSE cycle.
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        # Trim context if needed
        self._trim_context()

        # Make API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            max_tokens=500,
        )

        # Track token usage
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens

        # Get assistant response
        assistant_message = response.choices[0].message.content

        # Add to conversation history
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "messages": len(self.messages),
            "context_tokens": self._count_context_tokens(),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost": self._estimate_cost(),
        }

    def _estimate_cost(self) -> float:
        """Estimate cost in USD (GPT-4o-mini pricing as of late 2024)."""
        # Note: Check OpenAI pricing page for current rates
        # GPT-4o-mini: $0.075 per 1M input, $0.30 per 1M output
        input_cost = (self.total_input_tokens / 1_000_000) * 0.075
        output_cost = (self.total_output_tokens / 1_000_000) * 0.30
        return input_cost + output_cost


def run_interactive_chat():
    """Run an interactive chat session with real-time stats."""

    print("\n" + "=" * 60)
    print("LESSON 4: Interactive Chat Loop")
    print("=" * 60)

    print("""
Now let's run a live chat! This demonstrates the complete interaction loop.

Choose a persona to chat with:
""")

    for i, (key, persona) in enumerate(PERSONAS.items(), 1):
        print(f"  {i}. {persona['name']}")

    print("\n  (or type a number 1-4)")

    choice = input("\nYour choice: ").strip()

    try:
        idx = int(choice) - 1
        persona_key = list(PERSONAS.keys())[idx]
    except (ValueError, IndexError):
        persona_key = "friendly"

    selected_persona = PERSONAS[persona_key]

    print(f"\nStarting chat with: {selected_persona['name']}")
    print("Type 'quit' to exit, 'stats' for usage, 'switch' to change persona\n")
    print("─" * 50)

    session = ChatSession(selected_persona["system_prompt"])

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break

        if user_input.lower() == 'stats':
            stats = session.get_stats()
            print(f"\n📊 Session Statistics:")
            print(f"   Messages: {stats['messages']}")
            print(f"   Context tokens: {stats['context_tokens']}")
            print(f"   Total input tokens: {stats['total_input_tokens']}")
            print(f"   Total output tokens: {stats['total_output_tokens']}")
            print(f"   Estimated cost: ${stats['estimated_cost']:.6f}")
            continue

        if user_input.lower() == 'switch':
            print("\nSwitch to which persona?\n")
            for i, (key, persona) in enumerate(PERSONAS.items(), 1):
                print(f"  {i}. {persona['name']}")

            choice = input("\nYour choice: ").strip()
            try:
                idx = int(choice) - 1
                persona_key = list(PERSONAS.keys())[idx]
                selected_persona = PERSONAS[persona_key]
                session = ChatSession(selected_persona["system_prompt"])
                print(f"\nSwitched to: {selected_persona['name']}")
                print("(Conversation history cleared)")
                print("─" * 50)
            except (ValueError, IndexError):
                print("Invalid choice, keeping current persona.")
            continue

        spinner = Spinner("Calling OpenAI API")
        spinner.start()
        response = session.send_message(user_input)
        spinner.stop()
        print(f"🤖 {selected_persona['name']}: {response}")

        # Show inline stats and command hints
        stats = session.get_stats()
        print(f"\n  [{stats['messages']} messages · {stats['context_tokens']} tokens · ${stats['estimated_cost']:.6f}] 'switch' persona | 'stats' details | 'quit' exit")

    # Final stats
    stats = session.get_stats()
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"  Total messages exchanged: {stats['messages']}")
    print(f"  Total input tokens used: {stats['total_input_tokens']}")
    print(f"  Total output tokens used: {stats['total_output_tokens']}")
    print(f"  Estimated total cost: ${stats['estimated_cost']:.6f}")
    print("\nKey Takeaways:")
    print("  • Every message adds to the context (and cost)")
    print("  • System prompts define your product's personality")
    print("  • Token management is crucial for production systems")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the complete tutorial."""

    print("""
╔════════════════════════════════════════════════════════════╗
║     Tutorial 1: The Interaction Loop                       ║
║     Building the "Hello World" of AI Products              ║
╚════════════════════════════════════════════════════════════╝

This tutorial covers:
  • How tokens work (the currency of AI)
  • Message structure (system, user, assistant)
  • System prompts as product specs
  • Building an interactive chat loop

Let's begin!
""")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("Please create a .env file with your API key.")
        return

    input("Press Enter to start Lesson 1...")

    # Run through the lessons
    demonstrate_token_economy()
    demonstrate_message_structure()
    demonstrate_personas()
    run_interactive_chat()

    print("""
╔════════════════════════════════════════════════════════════╗
║     Tutorial Complete!                                      ║
╚════════════════════════════════════════════════════════════╝

What you learned:
  ✓ Tokens are the unit of work (and cost) for LLMs
  ✓ Conversations are lists of messages with roles
  ✓ System prompts define your AI product's behavior
  ✓ Context management is essential for production

Next Steps:
  → Tutorial 2: Tool Use - Give your AI the ability to take actions!
""")


if __name__ == "__main__":
    main()
