from pyexpat import model

from pydantic import BaseModel, Field
from enum import Enum
from agents import Agent
from _01_cognitive_workspace import TaskType
from setup_openai import model


class TaskRepresentation(BaseModel):
    task_type: TaskType
    extracted_entities: list[str] = Field(default_factory=list)
    complexity_estimate: float = 0.0
    ambiguities: list[str] = Field(default_factory=list)


perception_agent = Agent(
   model=model,
    name="Perception",
    instructions="""
    You are the perception module of a cognitive agent.
    Your job is to analyze a user query and produce a structured
    understanding of the task BEFORE any action is taken.

    For each query, determine:

    1. task_type: Is this a simple_lookup, multi_step, contradictory,
       ambiguous, or compositional problem?
    2. extracted_entities: What are the key nouns, concepts, or
       identifiers in the query?
    3. complexity_estimate: From 0.0 (trivial) to 1.0 (highly complex).
       Consider: number of steps needed, whether information must be
       combined from multiple sources, whether the query contains
       contradictions or ambiguity.
    4. ambiguities: List anything in the query that could be
       interpreted multiple ways.

    Be honest about complexity. A question that looks simple but
    requires cross-referencing is at least 0.5. A question containing
    "but" or "however" or "already tried" is likely contradictory
    and should be at least 0.6.

    You do NOT answer the user's question. You only analyze it.
    """,
    output_type=TaskRepresentation,
)