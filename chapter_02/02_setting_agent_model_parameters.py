from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings
from dotenv import load_dotenv
import os

# Initialise. 
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-3.1-flash-lite",
    openai_client=external_client
)


# Building the agent. 
instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"""

agent = Agent(
    name="Research Planner", 
    instructions=instructions, 
    model=model,
    model_settings=ModelSettings(
        temperature=1,
        max_tokens=150,
        top_p=1, 
        # frequency_penalty=0.5,
        # presence_penalty=0.5
    )
)

input = "learn about AI agents"

result = Runner.run_sync(
    agent, 
    input
)

print(result.final_output)