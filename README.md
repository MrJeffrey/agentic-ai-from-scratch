# Building Agentic AI from Scratch

A hands-on workshop for Technical Product Managers to learn the "first principles" of AI agent architecture. Move beyond prompt engineering to understand how agentic products actually work.

## What You'll Learn

By the end of this workshop, you'll understand:

- How LLMs process conversations (tokens, context windows, system prompts)
- How AI agents take actions through tool/function calling
- How to make AI decision-making transparent and auditable
- How to build personalized experiences with memory
- How to ground AI responses in your data (RAG)
- How to orchestrate multiple agents for complex workflows
- How to build safe, autonomous AI systems
- How to measure and guard agent quality with evals

## What You'll Build

| Tutorial | What You'll Build |
|----------|-------------------|
| **1. Interaction Loop** | Terminal chat interface with persona switching |
| **2. Tool Use** | E-commerce assistant with product search & orders |
| **3. Reasoning (ReAct)** | Customer service agent with visible thinking |
| **4. Memory & Context** | Personal assistant that remembers you |
| **5. RAG** | "Chat with your docs" knowledge base |
| **6. Multi-Agent** | Content creation pipeline with specialized agents |
| **7. Autonomy & Guardrails** | Safe goal-seeking autonomous agent |
| **8. LLM Evals** | Eval harness combining rules + LLM judge |

## Prerequisites

- **Python 3.10+** installed on your system
- **API Key** from OpenAI
- **Intermediate Python** - You can read code and understand APIs
- **Product Mindset** - Focus on user value, latency, and reliability

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MrJeffrey/agentic-ai-from-scratch.git
cd agentic-ai-from-scratch
```

### 2. Get Your API Key

Get your API key from https://platform.openai.com/api-keys

### 3. Run Your First Tutorial

```bash
cd 1-interaction-loop
./start.sh
```

## How the Start Scripts Work

Each tutorial includes a `start.sh` script that handles all the setup automatically. Just run it and you're ready to go:

```bash
cd <tutorial-folder>
./start.sh
```

The script will:
- **Create a virtual environment** - Isolates dependencies for each tutorial
- **Install dependencies** - Runs `pip install -r requirements.txt` automatically
- **Check for your `.env` file** - Creates a template if missing and prompts you to add your API key
- **Launch the tutorial** - Starts the interactive Python script

This means you don't need to manually manage virtual environments or install packages. Each tutorial is self-contained and ready to run with a single command.

If you prefer manual setup:
```bash
cd <tutorial-folder>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 <main_script>.py
```

## Tutorial Overview

### Tutorial 1: The Interaction Loop
**The "Hello World" of AI Products**

Learn the fundamentals: tokens (the currency of AI), message roles (system/user/assistant), and how system prompts define your product's personality.

```
You → [Token Economy] → [Message Structure] → [System Prompts] → Chat Interface
```

### Tutorial 2: Tool Use
**Giving AI "Hands" to Take Actions**

Connect your AI to business logic through function calling. Build an agent that can search products, check orders, and calculate shipping.

```
User Question → AI Decides → Call Tool → Get Result → AI Responds
```

### Tutorial 3: Reasoning (ReAct)
**The "Why" Behind the "What"**

Implement the ReAct pattern to make AI thinking visible. Essential for debugging, compliance, and building trust.

```
THOUGHT → ACTION → OBSERVATION → ANSWER
```

### Tutorial 4: Memory & Context
**Personalization That Persists**

Build user profiles that survive sessions. Learn summarization strategies to manage token costs while maintaining context.

```
Short-term Context → Long-term Memory → User Profile → Personalized Experience
```

### Tutorial 5: RAG (Retrieval Augmented Generation)
**Chat with Your Data**

Ground AI responses in your actual documents. Reduce hallucinations and enable source citations.

```
Question → Semantic Search → Context Injection → Grounded Answer
```

### Tutorial 6: Multi-Agent Orchestration
**Teams of Specialized Agents**

Build a content creation pipeline with Planner, Writer, and Editor agents. Learn the Manager/Worker pattern.

```
Planner → Writer → Editor → Final Content
```

### Tutorial 7: Autonomy & Guardrails
**Safe Autonomous Systems**

Build goal-seeking agents with comprehensive safety limits: iteration caps, cost budgets, time limits, and human-in-the-loop checkpoints.

```
Goal → [Guardrails Check] → Execute → [Guardrails Check] → Complete/Escalate
```

### Tutorial 8: LLM Evals
**Prove Your Agent Works (and Stays That Way)**

Evaluate responses against product requirements using a JSON response contract, rule-based checks, and an LLM judge for nuance.

```
Question → JSON Answer → Rule Checks → LLM Judge → Pass/Fail + Cost
```

## Key Concepts Quick Reference

| Concept | What It Means | Why It Matters |
|---------|---------------|----------------|
| **Tokens** | Units LLMs use to process text (~4 chars) | Pricing and context limits |
| **Context Window** | Max tokens per conversation | Memory vs. cost trade-off |
| **System Prompt** | Instructions defining AI behavior | Your product spec |
| **Tool Calling** | AI invoking external functions | Actions, not just words |
| **ReAct** | Reasoning + Acting pattern | Transparent decision-making |
| **RAG** | Retrieval Augmented Generation | Ground AI in your data |
| **Guardrails** | Safety limits on autonomous agents | Prevent runaway costs/actions |
| **Evals** | Automated checks on model outputs | Catch regressions before users do |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OPENAI_API_KEY not found` | Run `./start.sh` to create a `.env` template, then add your API key |
| `Permission denied: start.sh` | Run `chmod +x start.sh` |
| Rate limit errors | Wait a minute or use a different API key |

