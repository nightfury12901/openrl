"""
Pydantic Models for the Legal Case Assistant OpenEnv Environment.

Defines the core data structures using `openenv-core` base classes:
- LegalAction (extends Action): Agent's response to a legal task
- LegalObservation (extends Observation): What the agent sees (task prompt + feedback) + base reward/done
- LegalEnvironmentState (extends State): Full episode tracking state
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from openenv.core.env_server import Action, Observation, State

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LegalReward(BaseModel):
    """Detailed reward breakdown for a single step (embedded in Observation or metadata)."""
    total: float
    structural_score: float
    content_score: float
    length_penalty: float
    repetition_penalty: float
    step_bonus: float
    detail: str


class LegalAction(Action):
    """Action submitted by the agent in response to a legal task."""
    response: str = Field(
        ...,
        description="The agent's complete response to the legal task.",
    )


class LegalObservation(Observation):
    """Observation presented to the agent. Includes base Observation fields (done, reward, metadata)."""
    task_id: str = Field(
        ...,
        description="Unique identifier for the current task (e.g. 'task_1').",
    )
    task_type: str = Field(
        ...,
        description="Type of legal task: 'classification', 'risk_detection', or 'clause_optimization'.",
    )
    difficulty: str = Field(
        ...,
        description="Task difficulty: 'easy', 'medium', or 'hard'.",
    )
    prompt: str = Field(
        ...,
        description="The full task prompt presented to the agent.",
    )
    input_text: str = Field(
        ...,
        description="The raw legal text (case description, contract clause, etc.) to analyse.",
    )
    expected_output_fields: List[str] = Field(
        default_factory=list,
        description="List of field names the agent's response MUST contain.",
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Grader feedback from the previous step (None on first step).",
    )
    step_number: int = Field(
        default=0,
        description="Current step number within this task.",
    )
    max_steps: int = Field(
        default=3,
        description="Maximum number of steps allowed for this task.",
    )
    reward_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description="Detailed breakdown of the reward components.",
    )


class LegalEnvironmentState(State):
    """Full state of the environment episode (inherits episode_id and step_count)."""
    current_task_index: int = Field(
        default=0,
        description="Index of the current task (0-based).",
    )
    total_steps: int = Field(
        default=0,
        description="Total steps taken across all tasks in the episode.",
    )
    task_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of task_id → best score achieved.",
    )
    task_feedbacks: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of task_id → latest feedback string.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full action/reward history for the episode.",
    )
    previous_responses: List[str] = Field(
        default_factory=list,
        description="List of prior responses for loop detection.",
    )
