from openai import OpenAI
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

# Initialise OPENAI to use Gemini API
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


def get_embedding(text, model="gemini-embedding-2"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# Sample documents
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

# Generate embeddings for each documents
embeddings = [get_embedding(doc) for doc in documents]

embeddings_arrays = np.array(embeddings)
print(embeddings_arrays.shape)

# Apply PCA to reduce dimension to 3
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings_arrays)

# Creating a 3D plot using Plotly
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            z=reduced_embeddings[:, 2],
            mode="markers+text",
            text=documents,
            hoverinfo="text",
            marker=dict(size=12, color=list(range(len(documents))), opacity=0.8),
        )
    ]
)

# Adding titles and labels to the plot
fig.update_layout(
    title="3D Plot of Document Embeddings",
    scene=dict(
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        zaxis_title="PCA Component 3",
    ),
)

fig.show()
