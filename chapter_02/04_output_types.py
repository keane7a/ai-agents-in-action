from pydantic import BaseModel, ConfigDict
from typing import Dict
from typing_extensions import TypedDict
from setup_openai import model
from agents import Agent, Runner

# Agent instruction
instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan in the format of Dict[int, str]
"""

# class ResearchPlanModel(BaseModel): 
#     tasks: Dict[int, str]
#     """A list of task to perform for research."""
    
# Example of Strict JSON schema is enabled, but the output type is not valid. 
# Either make the output type strict, or pass output_schema_strict=False to your Agent()
#
# agent = Agent(
#     name="Research Planner", 
#     instructions=instructions,
#     model=model,
#     output_type=ResearchPlanModel
# )


# Fixed output type error 

class Task(TypedDict): 
    id: int
    description: str

class ResearchPlanModel(BaseModel): 
    tasks: list[Task]
    """A list of task to perform for research."""
    
    model_config = ConfigDict(extra="forbid")


agent = Agent(
    name="Research Planner", 
    instructions=instructions,
    model=model,
    output_type=ResearchPlanModel
)

input = "learn about AI agents"

result = Runner.run_sync(
    agent, 
    input=input,
    )

print(result.final_output)