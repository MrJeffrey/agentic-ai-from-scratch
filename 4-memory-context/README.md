# Tutorial 4: User Context & Memory

**Duration: 30 minutes**

Learn how to create personalized AI experiences that remember users across sessions - the foundation of retention and engagement.

## What You'll Learn

- **Short-term Context**: Message history within a conversation
- **Long-term Memory**: Persistent information across sessions
- **User Profiles**: Structured preferences and facts
- **Summarization**: Compressing history to save tokens

## What You'll Build

- A persistent user profile system
- Conversation summarization engine
- A memory-enabled personal assistant
- An agent that learns about users over time

## Key Concepts

### 1. Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  CONVERSATION (Short-term)                                  │
│  • Message history in current session                       │
│  • Limited by context window                                │
│  • Lost when session ends                                   │
├─────────────────────────────────────────────────────────────┤
│  USER PROFILE (Long-term)                                   │
│  • Preferences: "prefers formal communication"              │
│  • Facts: "works in finance", "lives in NYC"                │
│  • History summary: "Last discussed laptop purchases"       │
├─────────────────────────────────────────────────────────────┤
│  SESSION STORAGE                                            │
│  • Database, file system, or key-value store                │
│  • Loaded at session start                                  │
│  • Updated during conversation                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Context Injection

User profile data is injected into the system prompt:

```python
system_prompt = f"""
You are a helpful assistant.

USER CONTEXT:
- Preferences: {user.preferences}
- Known facts: {user.facts}
- Last conversation: {user.summary}

Use this context to personalize responses.
"""
```

### 3. Summarization Strategy

When context grows too large:

```
[Msg 1] + [Msg 2] + [Msg 3] → SUMMARIZE → "Summary"
                              ↓
New context: [System] + [Summary] + [Msg 4] + [Msg 5]
```

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

1. **Memory Concepts** - Understanding different memory types
2. **User Profiles** - Creating and storing user data
3. **Summarization** - Compressing conversations
4. **Interactive Demo** - Chat with a memory-enabled agent

## Try These Interactions

| Say This | What Happens |
|----------|--------------|
| "I'm a morning person" | Stored as a fact |
| "I prefer casual conversation" | Stored as a preference |
| "profile" | View your stored profile |
| "stats" | View memory statistics |
| (Exit and restart) | Agent remembers you! |

## Code Walkthrough

### User Profile Class

```python
class UserProfile:
    def __init__(self, user_id: str):
        self.profile = {
            "preferences": {},
            "facts": [],
            "summary": None,
        }

    def add_preference(self, key, value):
        self.profile["preferences"][key] = value
        self.save()

    def get_context_string(self) -> str:
        # Returns formatted context for system prompt
```

### Memory Agent

```python
class MemoryAgent:
    def __init__(self, user_id: str):
        self.user_profile = UserProfile(user_id)
        self.system_prompt = self._build_personalized_prompt()

    def chat(self, user_input: str) -> str:
        # 1. Add message to context
        # 2. Manage context window (summarize if needed)
        # 3. Get response
        # 4. Extract and store user info
        # 5. Return response
```

### Information Extraction

```python
def _extract_user_info(self, message, response):
    # Use LLM to identify facts and preferences
    # Store them in the user profile
```

## Product Manager Insights

### Retention Through Memory

| Without Memory | With Memory |
|----------------|-------------|
| "Welcome!" | "Welcome back, Sarah!" |
| "What would you like?" | "Want to continue looking at laptops?" |
| Generic responses | Personalized recommendations |

### Privacy Considerations

Ask yourself:
- What should be remembered vs. forgotten?
- How long should memories persist?
- Can users delete their data?
- What's shared vs. private?

### Cost Optimization

| Strategy | Token Savings | Trade-off |
|----------|---------------|-----------|
| Summarize after N messages | ~60-70% | Some detail lost |
| Keep only recent messages | ~80% | Context lost |
| Profile injection | Minimal overhead | Only key facts |

## Exercises

1. **Add Memory Categories**: Separate work facts from personal facts
2. **Implement Forgetting**: Add an expiration time for memories
3. **Build a Dashboard**: Show users what the AI remembers
4. **Add Privacy Controls**: Let users delete specific memories

## Common Issues

| Problem | Solution |
|---------|----------|
| "Memories not persisting" | Check file permissions for JSON storage |
| "Too much context" | Lower `max_context_messages` |
| "Extraction failing" | Check JSON parsing in `_extract_user_info` |
| "Not personalized" | Ensure profile is loaded in system prompt |

## Next Steps

Ready to connect your AI to your own knowledge base?

**→ Tutorial 5: RAG (Retrieval Augmented Generation)**

---

## Files in This Tutorial

| File | Purpose |
|------|---------|
| `start.sh` | Setup script |
| `memory_agent.py` | Main tutorial with memory system |
| `requirements.txt` | Python dependencies |
| `user_*.json` | Stored user profiles (created at runtime) |
| `README.md` | This file |
