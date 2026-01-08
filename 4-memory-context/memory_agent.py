#!/usr/bin/env python3
"""
Tutorial 4: User Context & Memory
==================================

This tutorial teaches you how to create personalized AI experiences:
- Short-term context (within a conversation)
- Long-term memory (across sessions)
- User profiling and preferences
- Conversation summarization for cost optimization

Learning Objectives:
- Understand the difference between context and memory
- Build a persistent user profile system
- Implement conversation summarization
- Create personalized AI interactions
"""

import os
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

# Safe import for tiktoken - fallback to character estimation if not installed
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


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
# LESSON 1: Understanding Memory Types
# =============================================================================

def demonstrate_memory_concepts():
    """Explain the different types of AI memory."""

    print("\n" + "=" * 60)
    print("LESSON 1: Types of AI Memory")
    print("=" * 60)

    print("""
AI systems have different "memory" mechanisms:

┌─────────────────────────────────────────────────────────────────┐
│  SHORT-TERM CONTEXT                                             │
│  ─────────────────                                              │
│  • Lives within a single conversation                           │
│  • Stored in the message history sent to the API                │
│  • Lost when the conversation ends                              │
│  • Limited by context window (tokens)                           │
│                                                                 │
│  Example: "Earlier you said you liked pizza" (same chat)        │
├─────────────────────────────────────────────────────────────────┤
│  LONG-TERM MEMORY                                               │
│  ────────────────                                               │
│  • Persists across sessions                                     │
│  • Stored in a database or file system                          │
│  • Must be explicitly loaded and saved                          │
│  • You control what's remembered                                │
│                                                                 │
│  Example: "Welcome back! Last time you ordered a laptop."       │
├─────────────────────────────────────────────────────────────────┤
│  USER PROFILE                                                   │
│  ────────────                                                   │
│  • Structured data about the user                               │
│  • Preferences, history, status                                 │
│  • Injected into system prompt                                  │
│  • Updated based on interactions                                │
│                                                                 │
│  Example: "As a premium member who prefers morning deliveries"  │
└─────────────────────────────────────────────────────────────────┘

KEY INSIGHT: LLMs have no built-in memory. YOU must build it.
""")

    input("Press Enter to continue to the User Profile system...")


# =============================================================================
# LESSON 2: User Profile System
# =============================================================================

def _is_persistence_enabled() -> bool:
    """Check whether disk persistence is enabled (default ON)."""
    env_value = os.getenv("ENABLE_MEMORY_PERSIST")
    if env_value is None:
        return True  # default: persist to demonstrate long-term memory
    return env_value.lower() in ("1", "true", "yes", "on")


class UserProfile:
    """
    A persistent user profile for personalization.

    In production, this would be stored in a database.
    Here, we use a JSON file for simplicity.
    """

    def __init__(self, user_id: str, storage_dir: str = ".", persist: Optional[bool] = None):
        # Sanitize user_id to prevent path traversal attacks
        import re
        safe_user_id = re.sub(r'[^a-zA-Z0-9_-]', '', user_id)
        if not safe_user_id:
            safe_user_id = "anonymous"
        self.user_id = safe_user_id
        self.persist = _is_persistence_enabled() if persist is None else persist
        self.storage_path = os.path.join(storage_dir, f"user_{safe_user_id}.json") if self.persist else None
        self.profile = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Load existing profile or create a new one."""
        if self.persist and self.storage_path and os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                return json.load(f)

        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "preferences": {},
            "facts": [],
            "conversation_count": 0,
            "last_interaction": None,
            "summary": None,
        }

    def save(self):
        """Persist the profile to storage."""
        self.profile["last_interaction"] = datetime.now().isoformat()
        if not self.persist or not self.storage_path:
            return
        with open(self.storage_path, 'w') as f:
            json.dump(self.profile, f, indent=2)

    def add_preference(self, key: str, value: str):
        """Add or update a user preference."""
        self.profile["preferences"][key] = value
        self.save()

    def add_fact(self, fact: str):
        """Add a fact about the user."""
        if fact not in self.profile["facts"]:
            self.profile["facts"].append(fact)
            # Keep only the most recent 10 facts
            self.profile["facts"] = self.profile["facts"][-10:]
            self.save()

    def increment_conversation(self):
        """Track conversation count."""
        self.profile["conversation_count"] += 1
        self.save()

    def set_summary(self, summary: str):
        """Set a summary of past interactions."""
        self.profile["summary"] = summary
        self.save()

    def get_context_string(self) -> str:
        """Generate a context string for the system prompt."""
        parts = []

        if self.profile["preferences"]:
            prefs = ", ".join(f"{k}: {v}" for k, v in self.profile["preferences"].items())
            parts.append(f"User preferences: {prefs}")

        if self.profile["facts"]:
            facts = "; ".join(self.profile["facts"])
            parts.append(f"Known facts about user: {facts}")

        if self.profile["summary"]:
            parts.append(f"Previous conversation summary: {self.profile['summary']}")

        if self.profile["conversation_count"] > 1:
            parts.append(f"This is conversation #{self.profile['conversation_count']} with this user")

        return "\n".join(parts) if parts else "No prior information about this user."

    def display(self):
        """Display the current profile."""
        print("\n📋 User Profile:")
        print(f"   ID: {self.profile['user_id']}")
        print(f"   Conversations: {self.profile['conversation_count']}")
        print(f"   Last seen: {self.profile.get('last_interaction', 'Never')}")
        print(f"   Preferences: {self.profile['preferences']}")
        print(f"   Facts: {self.profile['facts']}")
        if self.profile.get('summary'):
            print(f"   Summary: {self.profile['summary'][:100]}...")


def demonstrate_user_profiles():
    """Show how user profiles work."""

    print("\n" + "=" * 60)
    print("LESSON 2: User Profile System")
    print("=" * 60)

    print("""
User profiles store information that persists across sessions.

Let's create and populate a profile...
""")

    # Create a demo profile
    profile = UserProfile("demo_user")

    print("Creating new user profile...")
    profile.add_preference("communication_style", "casual")
    profile.add_preference("timezone", "PST")
    profile.add_fact("Works in product management")
    profile.add_fact("Interested in AI and machine learning")
    profile.add_fact("Prefers concise responses")
    profile.increment_conversation()

    profile.display()
    if profile.persist:
        print("\n💾 Persistence is ON. Profile is stored on disk. Avoid saving sensitive/PII in demos.")
        print("Disable persistence with ENABLE_MEMORY_PERSIST=false.")
    else:
        print("\n⚠️  Persistence is OFF (ENABLE_MEMORY_PERSIST=false). Data stays in memory only.")

    print("\nThis profile would be injected into the system prompt:")
    print("─" * 50)
    print(profile.get_context_string())
    print("─" * 50)

    input("\nPress Enter to continue to conversation memory...")


# =============================================================================
# LESSON 3: Conversation Summarization
# =============================================================================

def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    if not TIKTOKEN_AVAILABLE:
        return len(text) // 4  # Rough estimate: ~4 chars per token
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def summarize_conversation(messages: list) -> str:
    """
    Summarize a conversation to reduce token usage.

    This is key for long-running conversations or
    creating memory from past sessions.
    """
    # Build conversation text
    conversation_text = ""
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"] if isinstance(msg["content"], str) else str(msg)
        conversation_text += f"{role}: {content}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Summarize this conversation in 2-3 sentences.
Focus on:
- Key topics discussed
- Any decisions made or preferences expressed
- Important information the user shared

Be concise but capture the essence.""",
            },
            {"role": "user", "content": conversation_text},
        ],
        max_tokens=150,
    )

    return response.choices[0].message.content


def demonstrate_summarization():
    """Show how conversation summarization works."""

    print("\n" + "=" * 60)
    print("LESSON 3: Conversation Summarization")
    print("=" * 60)

    print("""
As conversations grow, they consume more tokens (= cost).
Summarization compresses old messages while preserving meaning.

Strategy:
┌─────────────────────────────────────────────────────────────┐
│  Message 1 ─┐                                               │
│  Message 2 ─┤──► SUMMARIZE ──► "Summary of earlier chat"    │
│  Message 3 ─┘                                               │
│                                                             │
│  Message 4  (keep recent)                                   │
│  Message 5  (keep recent)                                   │
│  Message 6  (current)                                       │
└─────────────────────────────────────────────────────────────┘
""")

    # Sample long conversation
    sample_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi! I'm looking for a new laptop for my job as a data scientist."},
        {"role": "assistant", "content": "I'd be happy to help! As a data scientist, you'll want to consider RAM (16GB minimum, 32GB preferred), a good CPU for computations, and possibly a GPU for machine learning tasks. What's your budget range?"},
        {"role": "user", "content": "I'm thinking around $1500-2000. I work mainly with Python and Jupyter notebooks."},
        {"role": "assistant", "content": "Great budget! For Python and Jupyter work, I'd recommend looking at laptops with at least an Intel i7 or AMD Ryzen 7 processor, 32GB RAM, and 512GB SSD. The Dell XPS 15 or MacBook Pro 14 would be excellent choices in that range."},
        {"role": "user", "content": "I prefer Windows actually. And I need good battery life since I travel a lot for conferences."},
        {"role": "assistant", "content": "Got it - Windows preference and portability are important. I'd suggest the Lenovo ThinkPad X1 Carbon or the Dell XPS 15. Both offer excellent battery life (10+ hours), are lightweight, and have great keyboards for coding. The ThinkPad is especially known for durability on the road."},
    ]

    # Count original tokens
    original_tokens = sum(count_tokens(m["content"]) for m in sample_conversation)
    print(f"Original conversation: {len(sample_conversation)} messages, ~{original_tokens} tokens")

    # Summarize
    print("\nSummarizing conversation...\n")
    summary = summarize_conversation(sample_conversation)

    summary_tokens = count_tokens(summary)
    print(f"Summary ({summary_tokens} tokens):")
    print("─" * 50)
    print(summary)
    print("─" * 50)

    savings = ((original_tokens - summary_tokens) / original_tokens) * 100
    print(f"\n✨ Token reduction: {savings:.0f}%")
    print("   This summary can be stored as long-term memory!")

    input("\nPress Enter to continue to the Memory Agent...")


# =============================================================================
# LESSON 4: Memory-Enabled Agent
# =============================================================================

class MemoryAgent:
    """
    An agent with both short-term context and long-term memory.

    This demonstrates the complete memory architecture.
    """

    def __init__(self, user_id: str, max_context_messages: int = 10):
        self.user_profile = UserProfile(user_id)
        self.user_profile.increment_conversation()
        self.max_context_messages = max_context_messages
        self.messages = []

        # Build system prompt with user context
        self.system_prompt = self._build_system_prompt()
        self.messages.append({"role": "system", "content": self.system_prompt})

    def _build_system_prompt(self) -> str:
        """Build a personalized system prompt."""
        base_prompt = """You are a helpful personal assistant with memory.

You remember information about the user from previous conversations.
Use this context to provide personalized, relevant responses.

When the user shares new information about themselves (preferences, facts, etc.),
acknowledge it and remember it for future reference.

Be warm and personable, referencing past interactions when relevant.
"""

        user_context = self.user_profile.get_context_string()

        return f"""{base_prompt}

USER CONTEXT:
{user_context}
"""

    def _extract_user_info(self, user_message: str, assistant_response: str):
        """
        Extract and store information about the user.

        In production, you might use structured extraction or
        a separate call to identify facts and preferences.

        NOTE: This makes a separate API call after every interaction, which
        doubles latency and cost. In production, run this asynchronously
        (e.g., via background task or queue) to avoid blocking the chat loop.
        """
        # Simple extraction using the LLM
        extraction_prompt = f"""Based on this exchange, extract any new facts or preferences about the user.
Return as JSON with keys "facts" (list of strings) and "preferences" (dict).
If nothing new, return empty lists/dicts.

User said: {user_message}
Assistant said: {assistant_response}

Return ONLY valid JSON, no explanation."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=200,
                response_format={"type": "json_object"},  # Enforce JSON output
            )

            result = json.loads(response.choices[0].message.content)

            for fact in result.get("facts", []):
                self.user_profile.add_fact(fact)

            for key, value in result.get("preferences", {}).items():
                self.user_profile.add_preference(key, value)

        except json.JSONDecodeError:
            # JSON parsing failed - LLM didn't return valid JSON
            pass
        except KeyError:
            # Missing expected keys in response
            pass

    def _manage_context(self):
        """Trim old messages and summarize if needed."""
        # Keep system prompt + recent messages
        # Need at least 6 messages: 1 system + 4 to keep + 1 to summarize
        if len(self.messages) > self.max_context_messages + 1 and len(self.messages) >= 6:
            # Summarize older messages (skip system prompt, keep last 4)
            old_messages = self.messages[1:-4]

            if old_messages:
                summary = summarize_conversation(old_messages)
                self.user_profile.set_summary(summary)

                # Replace old messages with summary
                summary_msg = {
                    "role": "system",
                    "content": f"[Summary of earlier conversation: {summary}]",
                }
                self.messages = [self.messages[0], summary_msg] + self.messages[-4:]

    def chat(self, user_input: str) -> str:
        """Process a user message and return response."""
        # Refresh system prompt with latest profile data (may have been updated)
        self.messages[0] = {"role": "system", "content": self._build_system_prompt()}

        self.messages.append({"role": "user", "content": user_input})

        # Manage context window
        self._manage_context()

        # Get response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            max_tokens=500,
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        # Extract and store user information
        self._extract_user_info(user_input, assistant_message)

        return assistant_message

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "messages_in_context": len(self.messages),
            "user_facts": len(self.user_profile.profile["facts"]),
            "user_preferences": len(self.user_profile.profile["preferences"]),
            "conversation_count": self.user_profile.profile["conversation_count"],
        }


def run_memory_demo():
    """Demonstrate the memory agent."""

    print("\n" + "=" * 60)
    print("LESSON 4: Memory Agent Demo")
    print("=" * 60)

    print("""
Let's interact with a memory-enabled agent.
It will remember what you tell it, even in future sessions!

Try sharing some personal info:
  • "I'm a morning person who loves coffee"
  • "I work in finance and prefer formal communication"
  • "I'm learning Spanish and interested in travel"

Type 'profile' to see stored information.
Type 'stats' to see memory statistics.
Type 'quit' to exit.
""")

    # Use a consistent user ID; persistence is ON by default for the demo
    agent = MemoryAgent("tutorial_user")

    # Check if returning user
    if agent.user_profile.profile["conversation_count"] > 1:
        print(f"🎉 Welcome back! This is conversation #{agent.user_profile.profile['conversation_count']}")
        if agent.user_profile.profile.get("summary"):
            print(f"📝 I remember from last time: {agent.user_profile.profile['summary'][:100]}...")
    else:
        print("👋 Welcome, new user!")

    if agent.user_profile.persist:
        print("💾 Persistence is ON. Data is written to disk; clear files if sharing this machine.")
        print("Disable by setting ENABLE_MEMORY_PERSIST=false.")
    else:
        print("⚠️  Persistence is OFF. Set ENABLE_MEMORY_PERSIST=true to save profiles to disk.")

    print("─" * 50)

    while True:
        user_input = input("\n👤 You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break

        if user_input.lower() == 'profile':
            agent.user_profile.display()
            continue

        if user_input.lower() == 'stats':
            stats = agent.get_stats()
            print("\n📊 Memory Stats:")
            print(f"   Messages in context: {stats['messages_in_context']}")
            print(f"   Stored facts: {stats['user_facts']}")
            print(f"   Stored preferences: {stats['user_preferences']}")
            print(f"   Total conversations: {stats['conversation_count']}")
            continue

        spinner = Spinner("Calling OpenAI API")
        spinner.start()
        response = agent.chat(user_input)
        spinner.stop()
        print(f"\n🤖 Assistant: {response}")

        # Show inline stats and command hints
        stats = agent.get_stats()
        print(f"\n  [{stats['messages_in_context']} msgs · {stats['user_facts']} facts] 'profile' | 'stats' | 'quit' exit")

    # Final profile
    print("\n" + "─" * 50)
    print("Session ended. Your profile has been saved!")
    agent.user_profile.display()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the complete tutorial."""

    print("""
╔════════════════════════════════════════════════════════════╗
║     Tutorial 4: User Context & Memory                      ║
║     Personalization & State Management                     ║
╚════════════════════════════════════════════════════════════╝

This tutorial covers:
  • Short-term context vs. long-term memory
  • User profile systems
  • Conversation summarization
  • Building personalized AI experiences

Let's make AI that remembers!
""")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("Please create a .env file with your API key.")
        return

    input("Press Enter to start...")

    demonstrate_memory_concepts()
    demonstrate_user_profiles()
    demonstrate_summarization()
    run_memory_demo()

    print("""
╔════════════════════════════════════════════════════════════╗
║     Tutorial Complete!                                      ║
╚════════════════════════════════════════════════════════════╝

What you learned:
  ✓ LLMs have no built-in memory - you must build it
  ✓ User profiles store preferences and facts
  ✓ Summarization reduces token costs
  ✓ Context injection personalizes responses

Product Manager Insights:
  • Memory = Personalization = Retention
  • Consider: What should be remembered? (privacy!)
  • Trade-off: More context = more cost
  • User profiles enable features like "continue where you left off"

Try running this tutorial again - the agent will remember you!

Next Steps:
  → Tutorial 5: RAG - Connect your AI to your knowledge base
""")


if __name__ == "__main__":
    main()
