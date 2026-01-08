#!/usr/bin/env python3
"""
Tutorial 5: RAG (Retrieval Augmented Generation)
=================================================

This tutorial teaches you how to connect AI to your proprietary data:
- Why RAG instead of fine-tuning
- Embedding and semantic search basics
- Building a document Q&A system
- Grounding responses to reduce hallucinations

Learning Objectives:
- Understand the RAG architecture
- Build a simple vector search system
- Create a "chat with your docs" feature
- Ground AI responses in source documents
"""

import os
import json
import sys
import threading
import time
import math
from pathlib import Path
from typing import Optional
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
# LESSON 1: Why RAG?
# =============================================================================

def explain_rag_vs_training():
    """Explain when to use RAG vs. fine-tuning."""

    print("\n" + "=" * 60)
    print("LESSON 1: Why RAG?")
    print("=" * 60)

    print("""
You want the AI to know about YOUR data. You have two options:

┌─────────────────────────────────────────────────────────────────┐
│  OPTION 1: FINE-TUNING                                          │
│  ─────────────────────                                          │
│  Train the model on your data                                   │
│                                                                 │
│  Pros:                                                          │
│  • Knowledge "baked into" the model                             │
│  • No retrieval latency                                         │
│                                                                 │
│  Cons:                                                          │
│  • Expensive ($$ and time)                                      │
│  • Data goes stale (need to retrain)                            │
│  • No source attribution                                        │
│  • Can still hallucinate                                        │
├─────────────────────────────────────────────────────────────────┤
│  OPTION 2: RAG (Retrieval Augmented Generation)                 │
│  ─────────────────────────────────────────────────              │
│  Give the model relevant context at query time                  │
│                                                                 │
│  Pros:                                                          │
│  • Always up-to-date (no retraining)                            │
│  • Source attribution (cite your sources!)                      │
│  • Cost-effective                                               │
│  • Reduces hallucinations                                       │
│                                                                 │
│  Cons:                                                          │
│  • Retrieval latency                                            │
│  • Quality depends on search                                    │
│  • Context window limits                                        │
└─────────────────────────────────────────────────────────────────┘

For most use cases, RAG is the right choice!

The RAG Pipeline:
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Query  │ ──► │  Search │ ──► │ Context │ ──► │   LLM   │
│         │     │  Docs   │     │Injection│     │ Answer  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                     ▲
                     │
              [Your Documents]
""")

    input("Press Enter to continue to embeddings...")


# =============================================================================
# LESSON 2: Embeddings and Semantic Search
# =============================================================================

def get_embedding(text: str) -> list:
    """
    Get the embedding vector for a piece of text.

    Embeddings convert text into numbers that capture meaning.
    Similar texts have similar embeddings.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors.

    1.0 = identical, 0.0 = unrelated, -1.0 = opposite
    """
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude_a = math.sqrt(sum(x * x for x in vec1))
    magnitude_b = math.sqrt(sum(x * x for x in vec2))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def demonstrate_embeddings():
    """Show how embeddings capture semantic meaning."""

    print("\n" + "=" * 60)
    print("LESSON 2: Embeddings & Semantic Search")
    print("=" * 60)

    print("""
Embeddings convert text to vectors that capture MEANING.

Similar meanings = Similar vectors
Different meanings = Different vectors

Let's see this in action...
""")

    # First, let's see what an embedding actually looks like
    print("Let's see what an embedding actually looks like...")
    sample_text = "Hello world"
    sample_embedding = get_embedding(sample_text)

    print(f"""
When we send "{sample_text}" to OpenAI's embedding API, we get back
a list of {len(sample_embedding):,} numbers:

  [{sample_embedding[0]:+.4f}, {sample_embedding[1]:+.4f}, {sample_embedding[2]:+.4f}, {sample_embedding[3]:+.4f}, ... {len(sample_embedding):,} total numbers]

Each number is between -1 and +1. This list of numbers IS the embedding.
Another name for "list of numbers" is a VECTOR.

Think of it like this:
─────────────────────────────────────────────────────────────────
  TEXT        →   API   →   VECTOR (list of numbers)
  "Hello"     →   📡    →   [0.12, -0.34, 0.56, ...]
  "Hi there"  →   📡    →   [0.11, -0.33, 0.55, ...]  ← Similar!
  "Buy stocks"→   📡    →   [-0.89, 0.12, -0.45, ...] ← Different!
─────────────────────────────────────────────────────────────────

The magic: texts with similar MEANING get similar NUMBERS.
That's how we can search by meaning instead of exact keywords!
""")

    input("Press Enter to see this in action...")

    # Test sentences
    sentences = [
        "The cat sat on the mat",
        "A feline rested on the rug",  # Similar meaning
        "Dogs are great pets",  # Related but different
        "The stock market crashed today",  # Unrelated
    ]

    print("\nComputing embeddings for test sentences...")
    embeddings = {s: get_embedding(s) for s in sentences}

    base_sentence = sentences[0]
    print(f"\nBase sentence: \"{base_sentence}\"\n")

    print("Similarities to base sentence:")
    print("─" * 50)

    for sentence in sentences[1:]:
        sim = cosine_similarity(embeddings[base_sentence], embeddings[sentence])
        bar = "█" * int(sim * 20)
        print(f"{sim:.3f} {bar} \"{sentence}\"")

    print("""
💡 Key Insight:
   - "feline on rug" is very similar to "cat on mat" (semantics!)
   - "dogs are pets" is somewhat related (animals)
   - "stock market" is unrelated
""")

    input("Press Enter to see how we calculate these scores...")

    # Explain how similarity is calculated
    print("""
HOW DO WE GET THESE SCORES?
═══════════════════════════════════════════════════════════════

We use "cosine similarity" - but don't worry about the math!
Here's the simple idea:

  VECTOR A: [0.2, 0.5, 0.1, ...]  (1,536 numbers for sentence A)
  VECTOR B: [0.2, 0.4, 0.1, ...]  (1,536 numbers for sentence B)
                ↓
         Compare each pair of numbers
                ↓
         Combine into ONE score

The score tells us: how much do these vectors "point the same way"?

  1.0  = Identical meaning (vectors point same direction)
  0.5  = Somewhat related
  0.0  = Completely unrelated (vectors are perpendicular)
 -1.0  = Opposite meaning (vectors point opposite directions)

Example from above:
─────────────────────────────────────────────────────────────────
  "The cat sat on the mat"   vs   "A feline rested on the rug"

  Both sentences have 1,536 numbers. We compare them and get: 0.85

  0.85 is close to 1.0, so these sentences have SIMILAR meaning!
  (Even though they share almost no words in common)
─────────────────────────────────────────────────────────────────

This is the power of semantic search - we find meaning, not keywords!
""")

    input("Press Enter to continue to the knowledge base...")


# =============================================================================
# LESSON 3: Building a Knowledge Base
# =============================================================================

# Sample company knowledge base (in production, load from files/database)
KNOWLEDGE_BASE = [
    {
        "id": "policy-001",
        "title": "Return Policy",
        "content": """Our return policy allows customers to return most items within 30 days of purchase for a full refund. Items must be in original condition with tags attached. Electronics have a 15-day return window. Final sale items cannot be returned. Refunds are processed within 5-7 business days after we receive the returned item. Original shipping costs are non-refundable unless the return is due to our error.""",
        "category": "policies",
    },
    {
        "id": "policy-002",
        "title": "Shipping Information",
        "content": """We offer three shipping options: Standard (5-7 business days, free on orders over $50), Express (2-3 business days, $9.99), and Overnight (next business day, $24.99). Orders placed before 2 PM EST ship same day. We ship to all 50 US states. International shipping is available to Canada and Mexico for an additional fee. Tracking information is sent via email once your order ships.""",
        "category": "shipping",
    },
    {
        "id": "product-001",
        "title": "TechPro Laptop X1",
        "content": """The TechPro Laptop X1 is our flagship laptop featuring a 15.6-inch 4K display, Intel Core i7-13700H processor, 32GB DDR5 RAM, and 1TB NVMe SSD. Battery life is up to 12 hours. It weighs 4.2 pounds and includes a backlit keyboard. The X1 is ideal for professionals and content creators. Price: $1,499. Available in Silver and Space Gray. Comes with a 2-year warranty.""",
        "category": "products",
    },
    {
        "id": "product-002",
        "title": "EcoSmart Water Bottle",
        "content": """The EcoSmart Water Bottle is made from 100% recycled ocean plastic. It holds 24 oz of liquid and features double-wall vacuum insulation to keep drinks cold for 24 hours or hot for 12 hours. The leak-proof lid has a built-in carrying loop. Available in Ocean Blue, Forest Green, and Sunset Orange. Price: $35. Dishwasher safe. Portion of proceeds supports ocean cleanup efforts.""",
        "category": "products",
    },
    {
        "id": "support-001",
        "title": "Account Management",
        "content": """To manage your account, log in at myaccount.example.com. You can update your email, password, and shipping addresses. To delete your account, contact support@example.com. Account deletion is processed within 30 days. All personal data is removed except for order history required for legal compliance. You can also manage newsletter subscriptions and notification preferences from your account settings.""",
        "category": "support",
    },
    {
        "id": "support-002",
        "title": "Payment Methods",
        "content": """We accept Visa, Mastercard, American Express, Discover, PayPal, and Apple Pay. All transactions are secured with 256-bit SSL encryption. We do not store credit card numbers on our servers. For subscription products, your card will be charged automatically on the billing date. You can update payment methods in your account settings. Gift cards can be combined with other payment methods.""",
        "category": "support",
    },
    {
        "id": "faq-001",
        "title": "Warranty Information",
        "content": """All electronics come with a minimum 1-year manufacturer warranty. Extended warranties are available for purchase. The warranty covers defects in materials and workmanship but does not cover damage from accidents, misuse, or normal wear. To make a warranty claim, contact support with your order number and description of the issue. Warranty repairs typically take 2-3 weeks.""",
        "category": "faq",
    },
    {
        "id": "company-001",
        "title": "About Our Company",
        "content": """Founded in 2015, TechShop has grown to serve over 1 million customers worldwide. Our mission is to provide high-quality technology products at fair prices. We are committed to sustainability and have reduced our carbon footprint by 40% since 2020. Our headquarters is in San Francisco, with distribution centers in Nevada, Texas, and New Jersey. We employ over 500 people and have been recognized as a Best Place to Work.""",
        "category": "company",
    },
]


class SimpleVectorStore:
    """
    A simple in-memory vector store for RAG.

    In production, use Pinecone, Weaviate, Chroma, etc.
    """

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_document(self, doc: dict):
        """Add a document to the store."""
        # Create searchable text from title and content
        text = f"{doc['title']}: {doc['content']}"
        embedding = get_embedding(text)

        self.documents.append(doc)
        self.embeddings.append(embedding)

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Search for relevant documents.

        Returns top_k most similar documents.
        """
        query_embedding = get_embedding(query)

        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, i))

        # Sort by similarity (highest first)
        similarities.sort(reverse=True)

        # Return top_k results
        results = []
        for sim, idx in similarities[:top_k]:
            doc = self.documents[idx].copy()
            doc["similarity"] = round(sim, 3)
            results.append(doc)

        return results


def demonstrate_knowledge_base():
    """Show how the knowledge base and retrieval work."""

    print("\n" + "=" * 60)
    print("LESSON 3: Building a Knowledge Base")
    print("=" * 60)

    print("""
A knowledge base is your collection of documents that the AI can search.

We'll:
1. Load documents into a vector store
2. Compute embeddings for each document
3. Search using natural language queries
Note: Embeddings are paid API calls. Cache vectors to avoid re-embedding the same content.

Building the vector store...
""")

    # Build the vector store
    store = SimpleVectorStore()

    for doc in KNOWLEDGE_BASE:
        print(f"  Indexing: {doc['title']}")
        store.add_document(doc)

    print(f"\n✓ Indexed {len(KNOWLEDGE_BASE)} documents")

    # Test searches
    test_queries = [
        "How do I return something?",
        "Tell me about the laptop",
        "What payment types do you accept?",
    ]

    print("\n" + "─" * 50)
    print("Testing semantic search:")
    print("─" * 50)

    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        results = store.search(query, top_k=2)

        for i, doc in enumerate(results, 1):
            print(f"   {i}. [{doc['similarity']}] {doc['title']}")

    print("""

💡 Notice: The search understands meaning, not just keywords!
   - "return something" matches "Return Policy"
   - "laptop" matches "TechPro Laptop X1"
""")

    input("\nPress Enter to continue to the RAG agent...")

    return store


# =============================================================================
# LESSON 4: RAG Agent
# =============================================================================

class RAGAgent:
    """
    A complete RAG agent that answers questions from a knowledge base.
    """

    def __init__(self, vector_store: SimpleVectorStore):
        self.store = vector_store

    def answer(self, question: str, show_sources: bool = True) -> str:
        """
        Answer a question using RAG.

        1. Retrieve relevant documents
        2. Build context from documents
        3. Generate answer with citations
        """

        # Step 1: Retrieve relevant documents
        docs = self.store.search(question, top_k=3)

        if show_sources:
            print("\n📚 Retrieved sources:")
            for doc in docs:
                print(f"   • [{doc['similarity']}] {doc['title']}")

        # Step 2: Build context
        context_parts = []
        for doc in docs:
            context_parts.append(f"[Source: {doc['title']}]\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Step 3: Generate answer
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
1. Only use information from the context below
2. If the context doesn't contain the answer, say "I don't have information about that in my knowledge base"
3. Cite your sources by mentioning the document title
4. Be concise but complete
5. Never make up information not in the context

CONTEXT:
""" + context,
                },
                {"role": "user", "content": question},
            ],
            max_tokens=500,
        )

        return response.choices[0].message.content


def run_rag_demo(store: SimpleVectorStore):
    """Interactive RAG demonstration."""

    print("\n" + "=" * 60)
    print("LESSON 4: RAG Agent in Action")
    print("=" * 60)

    print("""
Now let's use the complete RAG pipeline!

The agent will:
1. Search for relevant documents
2. Use them as context
3. Generate a grounded answer

Try questions like:
  • "What's your return policy for electronics?"
  • "Tell me about the TechPro laptop specs"
  • "How can I delete my account?"
  • "Do you ship internationally?"

Type 'quit' to exit.
""")

    agent = RAGAgent(store)
    query_count = 0

    while True:
        user_input = input("\n👤 Question: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break

        query_count += 1
        answer = agent.answer(user_input, show_sources=True)
        print(f"\n🤖 Answer: {answer}")

        # Show inline stats and command hints
        print(f"\n  [{query_count} queries] 'quit' exit")


# =============================================================================
# LESSON 5: Grounding and Hallucination Prevention
# =============================================================================

def demonstrate_grounding():
    """Show how RAG prevents hallucinations."""

    print("\n" + "=" * 60)
    print("LESSON 5: Grounding & Hallucination Prevention")
    print("=" * 60)

    print("""
Without grounding, LLMs can "hallucinate" - making up plausible-sounding
but incorrect information.

RAG prevents this by:
1. Limiting the AI to your actual documents
2. Requiring source attribution
3. Instructing it to say "I don't know" when info isn't available

Let's compare...
""")

    question = "What is the battery life of your laptop?"

    # Without RAG (general knowledge)
    print("─" * 50)
    print("WITHOUT RAG (general knowledge):")
    print("─" * 50)

    ungrounded = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for TechShop, an electronics store.",
            },
            {"role": "user", "content": question},
        ],
        max_tokens=150,
    )
    print(f"Q: {question}")
    print(f"A: {ungrounded.choices[0].message.content}")
    print("\n⚠️  This answer might be made up!")

    # With RAG
    print("\n" + "─" * 50)
    print("WITH RAG (grounded in documents):")
    print("─" * 50)

    # Get the laptop document
    laptop_doc = next(d for d in KNOWLEDGE_BASE if "Laptop" in d["title"])

    grounded = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Answer based ONLY on this document. Cite the source.
If not in the document, say "I don't have that information."

DOCUMENT:
[{laptop_doc['title']}]
{laptop_doc['content']}""",
            },
            {"role": "user", "content": question},
        ],
        max_tokens=150,
    )
    print(f"Q: {question}")
    print(f"A: {grounded.choices[0].message.content}")
    print("\n✓ This answer is grounded in actual product data!")

    # Ask about something not in the knowledge base
    print("\n" + "─" * 50)
    print("ASKING ABOUT UNKNOWN INFORMATION:")
    print("─" * 50)

    unknown_q = "What colors does your gaming mouse come in?"

    grounded_unknown = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Answer based ONLY on this document.
If not in the document, say "I don't have that information in my knowledge base."

DOCUMENT:
[{laptop_doc['title']}]
{laptop_doc['content']}""",
            },
            {"role": "user", "content": unknown_q},
        ],
        max_tokens=100,
    )
    print(f"Q: {unknown_q}")
    print(f"A: {grounded_unknown.choices[0].message.content}")
    print("\n✓ Correctly admits it doesn't know!")

    input("\nPress Enter to continue to the interactive demo...")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the complete tutorial."""

    print("""
╔════════════════════════════════════════════════════════════╗
║     Tutorial 5: RAG (Retrieval Augmented Generation)       ║
║     Chat with Your Data                                    ║
╚════════════════════════════════════════════════════════════╝

This tutorial covers:
  • Why RAG vs. fine-tuning
  • Embeddings and semantic search
  • Building a knowledge base
  • Grounding to prevent hallucinations

Let's connect AI to your data!
""")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("Please create a .env file with your API key.")
        return

    input("Press Enter to start...")

    explain_rag_vs_training()
    demonstrate_embeddings()
    store = demonstrate_knowledge_base()
    demonstrate_grounding()
    run_rag_demo(store)

    print("""
╔════════════════════════════════════════════════════════════╗
║     Tutorial Complete!                                      ║
╚════════════════════════════════════════════════════════════╝

What you learned:
  ✓ RAG lets AI answer from your documents
  ✓ Embeddings enable semantic search
  ✓ Context injection grounds responses
  ✓ Grounding reduces hallucinations

Product Manager Insights:
  • RAG is often better than fine-tuning for knowledge
  • Search quality = Answer quality
  • Consider: What documents should be searchable?
  • Citations build user trust

Production Considerations:
  • Use a real vector database (Pinecone, Weaviate, Chroma)
  • Chunk large documents for better retrieval
  • Add metadata filtering (by date, category, etc.)

Next Steps:
  → Tutorial 6: Multi-Agent Orchestration
""")


if __name__ == "__main__":
    main()
