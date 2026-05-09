# Basic RAG Application with Hugging Face and ChromaDB

This document provides a detailed, line-by-line explanation of the RAG (Retrieval-Augmented Generation) implementation in `index.py`.

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [Complete Code](#complete-code)
3. [Step-by-Step Breakdown](#step-by-step-breakdown)
   - [Imports and Setup](#1-imports-and-setup)
   - [Environment Variables](#2-environment-variables)
   - [ChromaDB Setup](#3-chromadb-setup)
   - [Embedding Function](#4-embedding-function)
   - [Document Ingestion](#5-document-ingestion)
   - [Retrieval Logic](#6-retrieval-logic)
   - [Answer Generation](#7-answer-generation)
   - [Main Execution](#8-main-execution)
4. [How It All Works Together](#how-it-all-works-together)
5. [Running the Application](#running-the-application)

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that combines information retrieval with text generation to answer questions accurately. Instead of relying solely on what the model has memorized during training, RAG:

1. **Retrieves** relevant documents from a knowledge base
2. **Augments** the query with this retrieved context
3. **Generates** an answer based on the provided context

```
User Query → Retrieve Context → Generate Answer → Response
```

This approach reduces hallucinations and allows your AI to answer questions about your own data.

---

## Complete Code

```python
import os
import re
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Setup ChromaDB and Embedding Function
client = chromadb.PersistentClient(path="./chroma_db")
hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create a collection
collection = client.get_or_create_collection(name="my_docs", embedding_function=hf_ef)

# 2. Ingest Data
documents = [
    "The capital of France is Paris.",
    "The Pyramids of Giza are located in Egypt.",
    "The Great Wall of China is one of the seven wonders of the world.",
    "The project 'Apollo 11' landed the first humans on the Moon in 1969."
]
metadatas = [{"source": "geography"}, {"source": "history"}, {"source": "history"}, {"source": "space"}]
ids = ["id1", "id2", "id3", "id4"]

# Check if collection already has data to avoid duplicates
existing = collection.get()
if len(existing["ids"]) == 0:
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

# 3. Retrieval Logic
def retrieve_context(query, n_results=2):
    results = collection.query(query_texts=[query], n_results=n_results)
    if results['documents'] and results['documents'][0]:
        return "\n".join(results['documents'][0])
    return ""

# 4. Answer Extraction Logic (simple extractive QA from context)
def generate_answer(query, context):
    if not context:
        return "No relevant context found to answer your question."

    # Simple extractive approach: find the sentence that best answers the question
    sentences = re.split(r'[.!?]', context)
    sentences = [s.strip() for s in sentences if s.strip()]

    # For wh-questions, return the most relevant sentence
    query_words = set(query.lower().split())
    best_match = None
    best_score = 0

    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        # Calculate overlap score
        overlap = len(query_words & sentence_words)
        if overlap > best_score:
            best_score = overlap
            best_match = sentence

    return best_match if best_match else context

# 5. Run the Application
if __name__ == "__main__":
    user_query = "Where are the Pyramids located?"
    context = retrieve_context(user_query)
    answer = generate_answer(user_query, context)

    print(f"Query: {user_query}")
    print(f"Retrieved Context: {context}")
    print(f"AI Answer: {answer}")
```

---

## Step-by-Step Breakdown

### 1. Imports and Setup

```python
import os
import re
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
```

| Import | Purpose |
|--------|---------|
| `os` | Access operating system variables and paths |
| `re` | Regular expressions for text processing (splitting sentences) |
| `chromadb` | Vector database for storing and searching document embeddings |
| `embedding_functions` | Pre-built embedding functions from ChromaDB |
| `load_dotenv` | Load environment variables from a `.env` file |

---

### 2. Environment Variables

```python
load_dotenv()
```

**What it does:** Loads variables from a `.env` file into your system's environment variables.

**Why it matters:** API keys, tokens, and configuration values should never be hardcoded. The `.env` file (which should be in your `.gitignore`) stores secrets securely.

**Example `.env` file:**
```
HF_TOKEN=your_huggingface_token_here
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

---

### 3. ChromaDB Setup

```python
client = chromadb.PersistentClient(path="./chroma_db")
```

**What it does:** Creates a ChromaDB client that stores data persistently in the `./chroma_db` directory.

**Two types of clients:**
- `chromadb.Client()` - In-memory only (data lost when script ends)
- `chromadb.PersistentClient(path=...)` - Saves to disk (data persists)

**Why `PersistentClient`:** Your vector database survives script restarts, so you don't need to re-ingest documents every time.

---

### 4. Embedding Function

```python
hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**What it does:** Configures an embedding function that converts text into numerical vectors.

**What are embeddings?**
Embeddings are numerical representations of text where semantically similar texts have similar vectors. For example:
- "Where are the Pyramids?" → `[0.12, -0.45, 0.89, ...]`
- "Pyramids location" → `[0.11, -0.44, 0.88, ...]` (similar vector)

**Why `all-MiniLM-L6-v2`?**
- Small and fast (80MB)
- Good quality for general purposes
- Runs locally without API calls
- 384-dimensional output

**Alternative models:**
- `all-mpnet-base-v2` - Higher quality, slower
- `bge-large-en-v1.5` - State-of-the-art
- `text-embedding-ada-002` - OpenAI's model (requires API)

---

### 5. Create Collection

```python
collection = client.get_or_create_collection(name="my_docs", embedding_function=hf_ef)
```

**What it does:** Creates (or retrieves) a collection named "my_docs".

**Analogy:** A collection in ChromaDB is like a table in a SQL database. It holds your documents, their embeddings, and metadata.

**Parameters:**
| Parameter | Purpose |
|-----------|---------|
| `name` | Unique identifier for the collection |
| `embedding_function` | Automatically embeds documents and queries |

---

### 6. Document Ingestion

```python
documents = [
    "The capital of France is Paris.",
    "The Pyramids of Giza are located in Egypt.",
    "The Great Wall of China is one of the seven wonders of the world.",
    "The project 'Apollo 11' landed the first humans on the Moon in 1969."
]
metadatas = [{"source": "geography"}, {"source": "history"}, {"source": "history"}, {"source": "space"}]
ids = ["id1", "id2", "id3", "id4"]
```

**What it does:** Prepares data for insertion.

**Three required components:**

| Component | Description | Example |
|-----------|-------------|---------|
| `documents` | The actual text content | "Paris is the capital of France" |
| `metadatas` | Additional info for filtering | `{"source": "geography"}` |
| `ids` | Unique identifiers | `"doc_001"` |

**Why IDs matter:** ChromaDB requires unique IDs. If you try to add a document with an existing ID, it will throw an error.

---

### 7. Duplicate Prevention

```python
existing = collection.get()
if len(existing["ids"]) == 0:
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
```

**What it does:** Checks if the collection is empty before adding documents.

**Why this matters:**
- Without this check, running the script twice would cause an error (duplicate IDs)
- This pattern allows safe re-runs during development

**Alternative approaches:**
```python
# Option A: Try-except for duplicate handling
try:
    collection.add(...)
except Exception:
    pass  # Already exists

# Option B: Use collection.upsert() instead of add()
collection.upsert(...)  # Updates if ID exists, inserts if not
```

---

### 8. Retrieval Logic

```python
def retrieve_context(query, n_results=2):
    results = collection.query(query_texts=[query], n_results=n_results)
    if results['documents'] and results['documents'][0]:
        return "\n".join(results['documents'][0])
    return ""
```

**What it does:** Searches the vector database for documents similar to the query.

**How it works:**
1. The query text is embedded using the same embedding function
2. ChromaDB calculates cosine similarity between query and stored documents
3. Returns the top `n_results` most similar documents

**Return structure:**
```python
{
    'documents': [['Doc 1 text', 'Doc 2 text']],  # List of lists
    'metadatas': [[{'source': 'history'}, ...]],
    'ids': [['id2', 'id3']],
    'distances': [0.15, 0.32]  # Lower = more similar
}
```

**Why `results['documents'][0]`:** The outer list exists because you can batch multiple queries. `[0]` gets results for the first (and only) query.

---

### 9. Answer Generation (Extractive QA)

```python
def generate_answer(query, context):
    if not context:
        return "No relevant context found to answer your question."

    # Split context into sentences
    sentences = re.split(r'[.!?]', context)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Find best matching sentence
    query_words = set(query.lower().split())
    best_match = None
    best_score = 0

    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words & sentence_words)
        if overlap > best_score:
            best_score = overlap
            best_match = sentence

    return best_match if best_match else context
```

**What it does:** Extracts the most relevant sentence from the retrieved context.

**Step-by-step:**

1. **Check for empty context** - Return early if no context was retrieved
2. **Split into sentences** - Use regex to split on `.`, `!`, `?`
3. **Tokenize** - Convert query and sentences to word sets
4. **Calculate overlap** - Count shared words between query and each sentence
5. **Return best match** - The sentence with highest word overlap

**Example:**
```
Query: "Where are the Pyramids located?"
Query words: {'where', 'are', 'the', 'pyramids', 'located'}

Sentence 1: "The Pyramids of Giza are located in Egypt."
Words: {'the', 'pyramids', 'of', 'giza', 'are', 'located', 'in', 'egypt'}
Overlap: {'the', 'pyramids', 'are', 'located'} = 4 words ✓ BEST

Sentence 2: "The Great Wall of China is one of the seven wonders."
Words: {'the', 'great', 'wall', 'of', 'china', ...}
Overlap: {'the', 'of'} = 2 words
```

**Why extractive instead of generative?**
- No API key required
- Works offline
- Faster and cheaper
- No hallucination risk
- Perfect for factual QA

---

### 10. Main Execution

```python
if __name__ == "__main__":
    user_query = "Where are the Pyramids located?"
    context = retrieve_context(user_query)
    answer = generate_answer(user_query, context)

    print(f"Query: {user_query}")
    print(f"Retrieved Context: {context}")
    print(f"AI Answer: {answer}")
```

**What it does:** Orchestrates the RAG pipeline.

**Flow:**
```
1. Define user query
2. Retrieve relevant context from ChromaDB
3. Extract answer from context
4. Print results
```

**Sample output:**
```
Query: Where are the Pyramids located?
Retrieved Context: The Pyramids of Giza are located in Egypt.
The Great Wall of China is one of the seven wonders of the world.
AI Answer: The Pyramids of Giza are located in Egypt
```

---

## How It All Works Together

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INGESTION PHASE (runs once)                                 │
│     ┌──────────┐    ┌─────────────┐    ┌──────────────┐        │
│     │ Documents│ →  │  Embedding  │ →  │  ChromaDB    │        │
│     │  (text)  │    │   (vectors) │    │  (storage)   │        │
│     └──────────┘    └─────────────┘    └──────────────┘        │
│                                                                 │
│  2. QUERY PHASE (runs for each question)                        │
│     ┌──────────┐    ┌─────────────┐    ┌──────────────┐        │
│     │  Query   │ →  │   Search    │ →  │   Context    │        │
│     │  (text)  │    │  (similarity)│   │  (retrieved) │        │
│     └──────────┘    └─────────────┘    └──────────────┘        │
│                                                                 │
│  3. ANSWER PHASE                                                │
│     ┌──────────┐    ┌─────────────┐    ┌──────────────┐        │
│     │ Context  │ →  │   Extract   │ →  │    Answer    │        │
│     │ + Query  │    │  (matching) │    │   (output)   │        │
│     └──────────┘    └─────────────┘    └──────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Running the Application

### Prerequisites

1. **Install dependencies:**
```bash
pip install chromadb python-dotenv
```

2. **Create a `.env` file (optional):**
```
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

### Execute

```bash
cd Embedding
python index.py
```

### Expected Output

```
Query: Where are the Pyramids located?
Retrieved Context: The Pyramids of Giza are located in Egypt.
The Great Wall of China is one of the seven wonders of the world.
AI Answer: The Pyramids of Giza are located in Egypt
```

---

## Extending This Code

### Add More Documents

```python
documents = [
    # ... existing docs ...
    "Bangladesh gained independence in 1971.",
    "The population of Dhaka is over 21 million."
]
```

### Use a Generative LLM

Replace `generate_answer()` with an LLM call:

```python
from langchain_groq import ChatGroq

def generate_answer(query, context):
    llm = ChatGroq(model="llama-3.1-8b-instant")
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm.invoke(prompt)
    return response.content
```

### Add Metadata Filtering

```python
# Only search history documents
results = collection.query(
    query_texts=[query],
    n_results=2,
    where={"source": "history"}  # Filter condition
)
```

---

## Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| Embedding | Sentence Transformers | Convert text to vectors |
| Vector DB | ChromaDB | Store and search embeddings |
| Retrieval | Cosine Similarity | Find relevant documents |
| Generation | Extractive QA | Pull answers from context |

This implementation provides a foundation for building more advanced RAG systems with LangChain, multiple data sources, or cloud-based LLMs.
