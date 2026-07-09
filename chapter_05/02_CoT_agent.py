import asyncio 
from agents import Agent, Runner
from setup_openai import model 

cot_agent = Agent(
    model=model,
    name="TimeTravelerCoT", 
    instructions="""(
    "You are a time travel problem solver.", 
    "Work out the solution step by step, then give the final answer." 
    )"""
)

question = (
    "Starting in 2025, you travel 10 years to the past, the 5 years to the future."
    "What year did you end up in?"
)

result = asyncio.run(
    Runner.run(
        cot_agent, 
        input=question
    )
)

print(result.final_output)