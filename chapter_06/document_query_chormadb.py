from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import os

# Initialise OPENAI to use Gemini API
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

def get_embedding(text, model="gemini-embedding-2"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Sample documents
documents = [
    "The sky is blue and beautiful.",
    "Love this blue and beautiful sky!",
    "The quick brown fox jumps over the lazy dog.",
    "A king's breakfast has sausages, ham, bacon, eggs, toast, and beans",
    "I love green eggs, ham, sausages and bacon!",
    "The brown fox is quick and the blue dog is lazy!",
    "The sky is very blue and the sky is very beautiful today",
    "The dog is lazy but the brown fox is quick!"
]

# Generate embeddings
embeddings = [get_embedding(doc) for doc in documents]
ids = [f"id{i}" for i in range(len(documents))]

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("documents")

collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=ids
)

def query_chromadb(query, top_n=2): 
    """Return the text of the top_n results from chromaDB collection"""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )
    
    return [(id, score, text) for id, score, text in zip(results['ids'][0], results['distances'][0], results['documents'][0])]


while True: 
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    
    top_n = int(input("Enter the number of top results to retrieve: "))
    search_res = query_chromadb(query, top_n=top_n)
    for id, score, text in search_res:
        print(f"ID: {id}, Score: {score}, Text: {text}")
        
    print("\n")