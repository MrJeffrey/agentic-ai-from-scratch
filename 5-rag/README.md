# Tutorial 5: RAG (Retrieval Augmented Generation)

**Duration: 45 minutes**

Learn how to connect AI to your proprietary knowledge base - enabling "Chat with your Data" features without expensive fine-tuning.

## What You'll Learn

- **RAG vs Fine-tuning**: When to use each approach
- **Embeddings**: Converting text to semantic vectors
- **Semantic Search**: Finding relevant documents by meaning
- **Grounding**: Preventing hallucinations with source data

## What You'll Build

- A simple vector store for document search
- An embedding-based retrieval system
- A RAG agent that answers from documents
- A system that cites its sources

## Key Concepts

### 1. The RAG Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. USER QUERY                                              │
│     "What's your return policy?"                            │
│                          ↓                                  │
│  2. SEMANTIC SEARCH                                         │
│     Convert query to embedding → Find similar documents     │
│                          ↓                                  │
│  3. CONTEXT BUILDING                                        │
│     Inject retrieved documents into prompt                  │
│                          ↓                                  │
│  4. GROUNDED GENERATION                                     │
│     LLM answers based on provided context                   │
│                          ↓                                  │
│  5. CITED RESPONSE                                          │
│     "According to our Return Policy..."                     │
└─────────────────────────────────────────────────────────────┘
```

### 2. Embeddings

Embeddings convert text to vectors that capture meaning:

```python
# Similar meanings = Similar vectors
embed("cat sat on mat")    ≈ embed("feline on rug")     # High similarity
embed("cat sat on mat")    ≠ embed("stock market")      # Low similarity
```

### 3. RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Data freshness** | Always current | Stale until retrained |
| **Cost** | Low | High |
| **Setup time** | Hours | Days/weeks |
| **Source attribution** | Yes | No |
| **Hallucination risk** | Lower | Higher |
| **Best for** | Knowledge bases | Behavior/style changes |

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

1. **Why RAG?** - Understanding when to use RAG
2. **Embeddings Demo** - See semantic similarity in action
3. **Knowledge Base** - Build and search a document store
4. **Grounding Demo** - Compare grounded vs ungrounded responses
5. **Interactive RAG** - Chat with the knowledge base

## Sample Knowledge Base

The tutorial includes a mock company knowledge base with:
- Return and shipping policies
- Product information (laptop, water bottle)
- Account management guides
- Payment and warranty info
- Company background

## Try These Queries

| Query | Expected Behavior |
|-------|-------------------|
| "How do I return something?" | Retrieves Return Policy |
| "Tell me about the laptop" | Retrieves TechPro Laptop X1 |
| "What payment methods?" | Retrieves Payment Methods |
| "Do you have a smartphone?" | Says "I don't have that info" |

## Code Walkthrough

### Creating Embeddings

```python
def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding
```

### Simple Vector Store

```python
class SimpleVectorStore:
    def search(self, query: str, top_k: int = 3) -> list:
        query_embedding = get_embedding(query)

        # Calculate cosine similarity with all docs
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, i))

        # Return top_k most similar
        similarities.sort(reverse=True)
        return [self.documents[i] for _, i in similarities[:top_k]]
```

### RAG Answer Generation

```python
def answer(self, question: str) -> str:
    # 1. Retrieve relevant docs
    docs = self.store.search(question, top_k=3)

    # 2. Build context
    context = "\n\n".join([d["content"] for d in docs])

    # 3. Generate grounded answer
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"Answer ONLY from this context:\n{context}"},
            {"role": "user", "content": question}
        ]
    )
```

## Product Manager Insights

### When to Use RAG

**Use RAG for:**
- Company knowledge bases
- Product documentation
- FAQ systems
- Legal/compliance documents
- Any data that changes frequently

**Consider fine-tuning for:**
- Brand voice/style
- Specialized terminology
- Task-specific behavior

### Quality Levers

| Lever | Impact |
|-------|--------|
| Document quality | Garbage in, garbage out |
| Chunk size | Too big = noise, too small = missing context |
| top_k results | More = comprehensive, fewer = focused |
| Embedding model | Better model = better search |

### Trust-Building Features

- Show which sources were used
- Include confidence scores
- Let users verify claims
- Highlight direct quotes

## Exercises

1. **Add Your Own Documents**: Replace the knowledge base with your content
2. **Implement Chunking**: Split large documents into paragraphs
3. **Add Metadata Filtering**: Filter by category before search
4. **Build a Source Viewer**: Show exact quotes used

## Production Considerations

For production RAG systems, consider:

- **Vector Databases**: Pinecone, Weaviate, Chroma, Qdrant
- **Document Loaders**: LangChain, LlamaIndex
- **Chunking Strategies**: Semantic, sentence, paragraph
- **Hybrid Search**: Combine keyword + semantic search
- **Re-ranking**: Use a second model to rank results

## Common Issues

| Problem | Solution |
|---------|----------|
| "Irrelevant results" | Improve document chunking |
| "Partial information" | Increase top_k |
| "Still hallucinating" | Strengthen system prompt |
| "Slow retrieval" | Use a vector database |

## Next Steps

Ready to orchestrate multiple agents working together?

**→ Tutorial 6: Workflow Orchestration (Multi-Agent)**

---

## Files in This Tutorial

| File | Purpose |
|------|---------|
| `start.sh` | Setup script |
| `rag_agent.py` | Main tutorial with RAG implementation |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
