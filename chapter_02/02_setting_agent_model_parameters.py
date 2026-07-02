from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings
from dotenv import load_dotenv
from setup_openai import model


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