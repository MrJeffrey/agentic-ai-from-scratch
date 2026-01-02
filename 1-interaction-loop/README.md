# Tutorial 1: The Interaction Loop

**Duration: 30 minutes**

The "Hello World" of AI Agent development. This tutorial teaches you the fundamental building blocks of conversational AI products.

## What You'll Learn

- **The Request/Response Lifecycle**: How messages flow between users and AI
- **Token Economy**: Understanding costs, latency, and context limits
- **System Prompts**: How they define your product's persona and constraints
- **Context Management**: Keeping conversations within token budgets

## What You'll Build

- A functional terminal-based chat interface
- A token counter and cost estimator
- Multiple persona demonstrations
- A context-aware conversation manager

## Key Concepts

### 1. Tokens: The Currency of AI

Tokens are how LLMs process text:
- ~4 characters = 1 token (in English)
- You pay for both input AND output tokens
- Context windows limit total conversation length

### 2. Message Roles

Every conversation has three types of messages:

| Role | Purpose | Example |
|------|---------|---------|
| **system** | Define AI behavior, rules, persona | "You are a helpful pizza shop assistant" |
| **user** | Human input | "What toppings do you have?" |
| **assistant** | AI responses | "We have pepperoni, mushrooms, and olives!" |

### 3. System Prompts = Product Specs

Your system prompt IS your product definition:
- **Persona**: Who is the AI?
- **Constraints**: What can/can't it discuss?
- **Style**: Formal? Casual? Technical?
- **Rules**: Specific behaviors to enforce

## Getting Started

### 1. Run the Setup Script

```bash
chmod +x start.sh
./start.sh
```

This will:
- Create a virtual environment
- Install dependencies
- Check for your `.env` file
- Launch the tutorial

### 2. Configure Your API Key

Edit the `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-key-here
```

Get your key at: https://platform.openai.com/api-keys

### 3. Run the Tutorial

```bash
./start.sh
```

Or manually:

```bash
source venv/bin/activate
python chat_loop.py
```

## Tutorial Structure

The tutorial runs through 4 interactive lessons:

1. **Token Economy** - See how text converts to tokens
2. **Message Structure** - Understand the conversation format
3. **Persona Demo** - Same question, 4 different AI personalities
4. **Interactive Chat** - Build and use a real chat loop

## Code Walkthrough

### The ChatSession Class

```python
class ChatSession:
    def __init__(self, system_prompt, max_context_tokens=4000):
        self.messages = [{"role": "system", "content": system_prompt}]

    def send_message(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(...)
        self.messages.append({"role": "assistant", "content": response})
```

This pattern (accumulating messages) is the foundation of ALL conversational AI.

## Product Manager Insights

### Cost Considerations
- More context = higher cost per request
- Longer responses = more output tokens
- GPT-4o-mini is ~100x cheaper than GPT-4

### Latency Considerations
- Time to First Token (TTFT): ~200-500ms
- Token generation: ~30-50 tokens/second
- Longer context = slightly slower responses

### Design Decisions
- When to truncate history vs. summarize?
- What belongs in system prompt vs. user context?
- How to handle token limit errors gracefully?

## Exercises

1. **Modify the Pirate Persona**: Add constraints (e.g., "never discuss modern technology")
2. **Add a Token Budget**: Stop the conversation when cost exceeds $0.01
3. **Implement Summarization**: Summarize old messages instead of deleting them

## Common Issues

| Problem | Solution |
|---------|----------|
| "API key not found" | Check your `.env` file exists and has the key |
| "Rate limit exceeded" | Wait a minute, or use a different API key |
| "Context length exceeded" | Reduce `max_context_tokens` or summarize history |

## Next Steps

Ready to give your AI the ability to take actions? Head to:

**→ Tutorial 2: Tool Use (Connecting to Business Logic)**

---

## Files in This Tutorial

| File | Purpose |
|------|---------|
| `start.sh` | Setup script (venv, deps, env check) |
| `chat_loop.py` | Main tutorial code |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
