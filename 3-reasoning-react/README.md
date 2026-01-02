# Tutorial 3: Reasoning & Transparency (ReAct Pattern)

**Duration: 45 minutes**

Learn how to make AI decision-making transparent and auditable using the ReAct pattern - the foundation of reliable, debuggable AI systems.

## What You'll Learn

- **Chain of Thought (CoT)**: Why "thinking out loud" improves accuracy
- **The ReAct Pattern**: Reasoning + Acting in a structured loop
- **Audit Trails**: Creating transparent AI decisions
- **Trade-offs**: Speed vs. accuracy in production systems

## What You'll Build

- A customer service agent with explicit reasoning
- Tools for policy lookup, eligibility checking, and calculations
- A comparison between direct and reasoning-based approaches
- An interactive system that shows its thought process

## Key Concepts

### 1. Chain of Thought (CoT)

Instead of jumping to an answer, CoT prompts the AI to think step-by-step:

```
WITHOUT CoT:
Q: If I have 3 apples and give away half, then buy 5 more, how many?
A: 6.5 apples

WITH CoT:
Q: If I have 3 apples and give away half, then buy 5 more, how many?
A: Let me work through this step by step:
   - Starting with 3 apples
   - Give away half: 3/2 = 1.5 apples given away
   - Remaining: 3 - 1.5 = 1.5 apples
   - Buy 5 more: 1.5 + 5 = 6.5 apples
   Answer: 6.5 apples
```

### 2. The ReAct Pattern

ReAct structures reasoning with actions:

```
🤔 THOUGHT: What does the customer need? What info do I need?
🔧 ACTION: Use tool to get accurate data
📋 OBSERVATION: Here's what the tool returned
🤔 THOUGHT: Do I have enough information?
💡 ANSWER: Here's my response based on the facts
```

### 3. Why This Matters

| Aspect | Direct Answer | ReAct Pattern |
|--------|---------------|---------------|
| **Speed** | Fast | Slower |
| **Accuracy** | May hallucinate | Grounded in facts |
| **Debugging** | Black box | Full trace available |
| **Compliance** | Difficult | Auditable |
| **Use Case** | Simple queries | Complex decisions |

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

1. **Why Reasoning Matters** - Compare direct vs. CoT responses
2. **ReAct in Action** - Watch the agent think through problems
3. **Approach Comparison** - Side-by-side direct vs. ReAct
4. **Interactive Session** - Try your own complex queries

## Example Queries to Try

| Query | What to Observe |
|-------|-----------------|
| "What are store hours on Sunday?" | Simple lookup, minimal reasoning |
| "I bought a laptop 20 days ago with receipt. Can I return it?" | Multi-step: check item type, days, receipt |
| "As an elite member spending $150, what are my points worth?" | Calculation + policy lookup |
| "Compare express vs overnight shipping for a $45 order" | Multiple lookups, comparison reasoning |

## Code Walkthrough

### The ReAct System Prompt

```python
REACT_SYSTEM_PROMPT = """
IMPORTANT: Think step-by-step before taking action:

1. THOUGHT: Analyze what the customer needs
2. ACTION: Use appropriate tool(s)
3. OBSERVATION: Review tool results
4. THOUGHT: Do I have enough info?
5. ANSWER: Provide clear response

Show your reasoning using:
🤔 THOUGHT: [reasoning]
🔧 ACTION: [tool and why]
📋 OBSERVATION: [what you learned]
💡 ANSWER: [response]
"""
```

### The ReAct Loop

```python
while iteration < max_iterations:
    response = client.chat.completions.create(
        messages=messages,
        tools=REACT_TOOLS,
    )

    if response.tool_calls:
        # Execute tools, add results
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append({"role": "tool", "content": result})
    else:
        # Reasoning complete, return answer
        return response.content
```

## Product Manager Insights

### When to Use ReAct

**Use ReAct for:**
- Customer-facing decisions that need to be explained
- Compliance-sensitive operations
- Complex multi-step queries
- Debugging and quality assurance

**Use Direct for:**
- Simple factual lookups
- Latency-critical applications
- High-volume, low-stakes queries

### Logging Strategies

```python
# Log the full reasoning trace
audit_log = {
    "query": user_query,
    "reasoning_steps": reasoning_trace,
    "tools_called": tool_calls,
    "final_answer": response,
    "timestamp": datetime.now(),
}
```

### User Experience Considerations

- Should users see the reasoning? (transparency vs. noise)
- How to display "thinking" without frustrating users?
- When to show progress vs. just the answer?

## Exercises

1. **Add Confidence Scores**: Have the agent express certainty levels
2. **Implement Fallback**: What if the agent can't find information?
3. **Create Audit Dashboard**: Log and display reasoning traces
4. **Add Human Escalation**: Detect when to escalate to human

## Common Issues

| Problem | Solution |
|---------|----------|
| "Agent not showing reasoning" | Check system prompt for THOUGHT/ACTION format |
| "Too verbose reasoning" | Add instruction to be concise in thoughts |
| "Wrong tool selected" | Improve tool descriptions |
| "Infinite loops" | Set max_iterations, add stop conditions |

## Next Steps

Ready to make your AI remember users across sessions?

**→ Tutorial 4: Memory & Context (User Personalization)**

---

## Files in This Tutorial

| File | Purpose |
|------|---------|
| `start.sh` | Setup script |
| `react_agent.py` | Main tutorial with ReAct customer service agent |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
