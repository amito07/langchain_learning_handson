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