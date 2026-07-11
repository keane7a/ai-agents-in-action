import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Documents
documents = [
    "The sky is blue and beautiful.",
    "Love this blue and beautiful sky!",
    "The quick brown fox jumps over the lazy dog.",
    "A king's breakfast has sausages, ham, bacon, eggs, toast, and beans",
    "I love green eggs, ham, sausages and bacon!",
    "The brown fox is quick and the blue dog is lazy!",
    "The sky is very blue and the sky is very beautiful today",
    "The dog is lazy but the brown fox is quick!",
]

vectoriser = TfidfVectorizer()
X = vectoriser.fit_transform(documents)
vector_database = X.toarray()


def cosine_similarity_search(query, database, vectoriser, top_n=5):
    query_vec = vectoriser.transform([query]).toarray()
    similarities = cosine_similarity(query_vec, database)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [(index, similarities[index]) for index in top_indices]


while True:
    query = input("Enter a query document or 'exit' to quit: ").strip()

    if query.lower() == "exit":
        break

    top_n = int(input("How many top matches do you want to see?"))
    results = cosine_similarity_search(query, vector_database, vectoriser, top_n)

    print(f"Top Matched Documents:")
    for idx, score in results:
        print(f"- {documents[idx]} (Score: {score:.4f})")

    print("\n")
