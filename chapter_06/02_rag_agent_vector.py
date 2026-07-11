import uuid
from pathlib import Path
import chromadb
import tiktoken
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from setup_openai import model

load_dotenv()

# Load script
script_text = Path("sample_document/back_to_the_future.txt").read_text(encoding="utf-8")


def simple_chunk(text, max_tokens=200):
    tokeniser = tiktoken.get_encoding("cl100k_base")
    words, chunk, chunks = text.split(), [], []

    for w in words:
        if len(tokeniser.encode(" ".join(chunk + [w]))) > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = [w]
        else:
            chunk.append(w)

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


docs = simple_chunk(script_text, max_tokens=200)

client = chromadb.PersistentClient(path="./chapter_06/chroma_script_store")
collection_name = "bttf_script"

collection = client.get_or_create_collection(collection_name)

if collection.count() == 0:
    collection.add(ids=[str(uuid.uuid4()) for _ in docs], documents=docs)


@function_tool
def search_script(query: str, top_k: int = 3) -> str:
    res = collection.query(query_texts=[query], n_results=top_k)
    if res and "documents" in res and res["documents"] and res["documents"][0]:
        return "\n\n".join(res["documents"][0])

    print("No relevant documents found.")
    return "No relevant documents found."


agent = Agent(
    model=model,
    name="Script Agent",
    instructions="""
        You answer questions about the movie *Back to the Future*.\n
        When needed, call the `search_script` tool to fetch passages,
        then cite or paraphrase them in your answer.
        Make sure your answers are grounded in the script text. 
    """,
    tools=[search_script],
)

query = "Where does Doc tell Mary to meet him, and at what time?"
result = Runner.run_sync(agent, query)
print("\n-- Answer --\n", result.final_output)

query = "What happens at 1:15AM?"
result = Runner.run_sync(agent, query)
print("\n-- Answer --\n", result.final_output)
