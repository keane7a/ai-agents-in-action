from pydantic import BaseModel
from typing import List
from setup_openai import model
from agents import Agent, Runner

# Agent instruction
instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"""

class ResearchPlanModel(BaseModel): 
    task: List[str]
    """A list of task to perform for research."""
    

agent = Agent(
    name="Research Planner", 
    instructions=instructions,
    output_type=ResearchPlanModel,
    model=model
    )

input = "learn about AI agents"

result = Runner.run_sync(
    agent, 
    input=input,
    )

print(result.final_output)