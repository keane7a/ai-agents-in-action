from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings
from dotenv import load_dotenv
import os

# Initialise.
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-3.1-flash-lite", openai_client=external_client
)
