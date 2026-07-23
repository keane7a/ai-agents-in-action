import asyncio
import json
from pydantic import BaseModel, Field
from enum import Enum
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from _01_cognitive_workspace import (
    AttentionSignal,
    CognitiveWorkspace,
)
from _02_perception_module import TaskRepresentation, perception_agent
from _03_planning_module import PlanOutput, planning_agent
from _04_execution_module import execution_agent
from _05_evaluation_module import EvaluationResult, evaluation_agent
from _06_attention_module import route_attention
from _07_memory_module import memory_agent


# Workspace update helpers
def update_workspace_perception(workspace, result):
    output = result.final_output
    workspace.task_type = output.task_type
    workspace.extracted_entities = output.extracted_entities
    workspace.complexity_estimate = output.complexity_estimate
    workspace.ambiguities = output.ambiguities
    workspace.steps_taken.append("PERCEIVE")
    print(
        f" [Perception] Type={output.task_type.value},"
        f"Complexity={output.complexity_estimate:.2f},"
        f"Entities={output.extracted_entities},"
    )


def update_workspace_memory(workspace, result):
    output = result.final_output
    if isinstance(output, str) and output.strip():
        workspace.memory_hits.append(output)
    workspace.steps_taken.append("MEMORY")
    print(f" [Memory] Hits={len(workspace.memory_hits)}")


def update_workspace_plan(workspace, result):
    output = result.final_output
    workspace.current_strategy = output.strategy_type
    workspace.sub_goals = output.sub_goals
    workspace.alternative_strategies = output.alternative_strategies
    workspace.steps_taken.append("PLAN")
    print(
        f" [Planning] Strategy={output.strategy_type.value},"
        f" Sub-goals={output.sub_goals},"
    )


def update_workspace_execution(workspace, result):
    output = result.final_output
    workspace.findings.append(output)
    if workspace.sub_goals:
        workspace.sub_goals.pop(0)
    print(
        f" [Execution] Found={len(workspace.findings)}"
        f"(relevance={output.relevance_score:.2f})"
    )


def update_workspace_evaluation(workspace, result):
    output = result.final_output
    workspace.confidence = max(
        0.0, min(1.0, workspace.confidence + output.confidence_delta)
    )
    workspace.confidence_trend.append(workspace.confidence)

    if not output.consistency_check:
        workspace.active_signal = AttentionSignal.CONTRADICTION
        workspace.signal_source = "evaluation"

    if output.recommendation == "REPLAN":
        workspace.active_signal = AttentionSignal.STAGNATION
        workspace.signal_source = "evaluation"
    elif output.recommendation == "ESCALATE":
        workspace.active_signal = AttentionSignal.LOW_CONFIDENCE
        workspace.signal_source = "evaluation"
    elif output.recommendation == "TERMINATE":
        workspace.active_signal = AttentionSignal.TASK_COMPLETE
        workspace.signal_source = "evaluation"

    workspace.steps_taken.append("EVALUATE")
    print(
        f" [Evaluation] Confidence={workspace.confidence:.2f},"
        f"Recommendation={output.recommendation},"
    )


# RESPONSE BUILDER
def build_response(workspace: CognitiveWorkspace) -> str:
    findings_text = "\n".join(
        f"- {f.content} (source: {f.source}, " f"relevance: {f.relevance_score:.2f})"
        for f in workspace.findings
    )

    return (
        f"Query: {workspace.raw_query}\n"
        f"Strategy: {workspace.current_strategy.value}\n"
        f"Confidence: {workspace.confidence:.2f}\n"
        f"Findings:\n{findings_text}\n"
        f"Steps Taken: {len(workspace.steps_taken)}\n"
    )


def build_uncertain_response(workspace: CognitiveWorkspace) -> str:
    return (
        f"Query: {workspace.raw_query}\n"
        f"NOTE: Low confidence ({workspace.confidence:.2f}). "
        f"The agent could not resolve this with sufficient certainty.\n"
        f"Partial findings:\n"
        + "\n".join(f"- {f.content}" for f in workspace.findings)
        + f"\nFailed approaches: {workspace.failed_approaches}\n"
        f"Suggestion: This query may require human expertise or "
        f"additional data sources."
    )


# Format helpers
def format_memory_query(workspace: CognitiveWorkspace) -> str:
    return (
        f"Search for past experience related to these entities: "
        f"{workspace.extracted_entities}. "
        f"Task type: {workspace.task_type.value}. "
        f"Query: {workspace.raw_query}"
    )


def format_planning_input(workspace: CognitiveWorkspace) -> str:
    return json.dumps(
        {
            "task_type": workspace.task_type.value,
            "entities": workspace.extracted_entities,
            "complexity": workspace.complexity_estimate,
            "ambiguities": workspace.ambiguities,
            "memory_hits": workspace.memory_hits,
            "failed_approaches": workspace.failed_approaches,
        }
    )


def format_execution_input(workspace: CognitiveWorkspace) -> str:
    current_goal = (
        workspace.sub_goals[0] if workspace.sub_goals else workspace.raw_query
    )
    return json.dumps(
        {
            "sub_goal": current_goal,
            "context": {
                "task": workspace.raw_query,
                "strategy": workspace.current_strategy.value,
                "prior_findings": [f.content[:100] for f in workspace.findings[-3:]],
            },
        }
    )


def format_evaluation_input(workspace: CognitiveWorkspace) -> str:
    return json.dumps(
        {
            "task": workspace.raw_query,
            "current_strategy": workspace.current_strategy.value,
            "latest_finding": (
                workspace.findings[-1].model_dump() if workspace.findings else None
            ),
            "all_findings_count": len(workspace.findings),
            "confidence": workspace.confidence,
            "confidence_trend": workspace.confidence_trend[-5:],
            "steps_taken": workspace.steps_taken[-5:],
            "remaining_sub_goals": workspace.sub_goals,
        }
    )


# Experience recording
async def record_experience(workspace: CognitiveWorkspace):
    summary = (
        f"Task: {workspace.raw_query}\n"
        f"Type: {workspace.task_type.value}\n"
        f"Strategy: {workspace.current_strategy.value}\n"
        f"Confidence: {workspace.confidence:.2f}\n"
        f"Steps: {len(workspace.steps_taken)}\n"
        f"Failed approaches: {workspace.failed_approaches}"
    )
    print(
        f"  [Memory] Recording experience: "
        f"{workspace.task_type.value} -> "
        f"{workspace.current_strategy.value} "
        f"(confidence={workspace.confidence:.2f})"
    )


# Cognitive Loop


async def run_cognitive_loop(
    query: str,
    max_iterations: int = 10,
    confidence_threshold: float = 0.8,
    max_cognitive_steps: int = 5,
    mcp_servers: list = None,
):
    workspace = CognitiveWorkspace(raw_query=query)

    # Attach MCP servers to execution agent if provided
    exec_agent = execution_agent
    if mcp_servers:
        exec_agent = execution_agent.clone(mcp_servers=mcp_servers)

    for i in range(max_iterations):
        workspace.iteration_count = i
        print(f"\n=== Iteration {i + 1} ===")

        # Inner cognitive cycle
        for step in range(max_cognitive_steps):
            next_module = route_attention(workspace)
            print(f"  Routing -> {next_module}")

            if next_module == "FAST_RESPOND":
                return build_response(workspace)

            if next_module == "RESPOND":
                return build_response(workspace)

            if next_module == "ESCALATE":
                return build_uncertain_response(workspace)

            if next_module == "PERCEIVE":
                result = await Runner.run(perception_agent, input=workspace.raw_query)
                update_workspace_perception(workspace, result)

            elif next_module in ("PLAN", "META_PLAN"):
                result = await Runner.run(
                    planning_agent, input=format_planning_input(workspace)
                )
                update_workspace_plan(workspace, result)

            elif next_module == "MEMORY":
                # Memory module runs without MCP server in this
                # standalone demo; in production, attach memory_server
                workspace.steps_taken.append("MEMORY")
                print("  [Memory] No MCP memory server in standalone mode")
                workspace.active_signal = AttentionSignal.NONE

            elif next_module == "EXECUTE":
                result = await Runner.run(
                    exec_agent, input=format_execution_input(workspace)
                )
                update_workspace_execution(workspace, result)

            elif next_module == "EVALUATE":
                result = await Runner.run(
                    evaluation_agent, input=format_evaluation_input(workspace)
                )
                update_workspace_evaluation(workspace, result)

                if workspace.active_signal == AttentionSignal.TASK_COMPLETE:
                    break

            # Reset signal after processing
            workspace.active_signal = AttentionSignal.NONE

        # Check convergence at the agentic loop level
        if workspace.confidence >= confidence_threshold:
            print(f"\nConverged at confidence {workspace.confidence:.2f}")
            break

    # Record experience in memory after completion
    await record_experience(workspace)
    return build_response(workspace)


async def main():
    query = (
        "A user reports that their deployment pipeline fails "
        "intermittently, but only during high-traffic periods. "
        "The standard fix (increasing timeout thresholds) was "
        "already applied and did not resolve the issue."
    )
    print(f"Query: {query}\n")

    response = await run_cognitive_loop(query)
    print(f"\n{'='*60}")
    print("FINAL RESPONSE:")
    print(response)


asyncio.run(main())
