from pydantic import BaseModel, Field
from enum import Enum
from _01_cognitive_workspace import StrategyType, AttentionSignal, CognitiveWorkspace


def route_attention(workspace: CognitiveWorkspace) -> str:
    """Read workspace signals and determine next module."""

    # System 1 fast path: bypass full cognitive cycle for simple,
    # familiar problems
    if (
        workspace.complexity_estimate < 0.3
        and workspace.memory_hits
        and workspace.active_signal == AttentionSignal.NONE
    ):
        return "FAST_RESPOND"

    # Signal-driven routing
    if workspace.active_signal == AttentionSignal.TASK_COMPLETE:
        return "RESPOND"

    if workspace.active_signal == AttentionSignal.STAGNATION:
        # Meta-planning records the failed strategy to prevent repetition
        workspace.failed_approaches.append(workspace.current_strategy.value)
        return "META_PLAN"

    if workspace.active_signal == AttentionSignal.CONTRADICTION:
        # Override the current strategy when contradictions are detected
        workspace.current_strategy = StrategyType.HYPOTHESIS_TEST
        return "PLAN"

    if workspace.active_signal == AttentionSignal.LOW_CONFIDENCE:
        if workspace.confidence < 0.2:
            # Below minimum confidence threshold, surface uncertainty
            return "ESCALATE"
        return "PLAN"

    if workspace.active_signal == AttentionSignal.KNOWLEDGE_GAP:
        return "MEMORY"

    # Default cognitive cycle
    if not workspace.extracted_entities:
        return "PERCEIVE"
    if not workspace.sub_goals:
        return "PLAN"
    if (
        workspace.iteration_count == 0
        or workspace.active_signal == AttentionSignal.NONE
    ):
        return "EXECUTE"

    return "EVALUATE"
