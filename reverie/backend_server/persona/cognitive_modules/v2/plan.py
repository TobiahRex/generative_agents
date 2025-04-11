"""
Plan module for generative agents.

This module handles planning and scheduling for generative agents, including:
- Daily planning
- Action planning
- Goal management
- Schedule management
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import random

from reverie.backend_server.persona.cognitive_modules.v2.associative_memory import (
    AssociativeMemory,
)
from reverie.backend_server.persona.cognitive_modules.v2.spatial_memory import (
    SpatialMemory,
)
from reverie.backend_server.persona.cognitive_modules.v2.scratch import Scratch

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """Represents an action in a plan."""

    name: str
    description: str
    start_time: datetime
    end_time: datetime
    location: Dict[str, Any]
    priority: float
    dependencies: List[str]
    status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "location": self.location,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "status": self.status,
        }


@dataclass
class Goal:
    """Represents a goal in a plan."""

    name: str
    description: str
    deadline: datetime
    priority: float
    progress: float
    status: str
    subgoals: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "deadline": self.deadline,
            "priority": self.priority,
            "progress": self.progress,
            "status": self.status,
            "subgoals": self.subgoals,
        }


class PlanGenerator:
    """Handles plan generation and optimization."""

    def __init__(self):
        self.time_slot_duration = timedelta(hours=1)
        self.max_actions_per_day = 8
        self.priority_weights = {
            "work": 1.0,
            "social": 0.8,
            "personal": 0.6,
            "leisure": 0.4,
        }

    def generate_daily_plan(
        self, persona: Any, maze: Any, start_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate a daily plan for the agent."""
        try:
            if start_time is None:
                start_time = datetime.now()

            # Get current state
            current_state = persona.scratch.get_current_state()

            # Generate goals
            goals = self._generate_goals(persona, current_state)

            # Generate actions
            actions = self._generate_actions(persona, goals, maze)

            # Optimize schedule
            schedule = self._optimize_schedule(actions, start_time)

            return {
                "goals": [goal.to_dict() for goal in goals],
                "actions": [action.to_dict() for action in actions],
                "schedule": schedule,
            }

        except Exception as e:
            logger.error(f"Error generating daily plan: {e}")
            return {}

    def _generate_goals(
        self, persona: Any, current_state: Dict[str, Any]
    ) -> List[Goal]:
        """Generate goals based on current state and persona."""
        try:
            goals = []

            # Get relevant memories
            memories = persona.a_mem.retrieve_memories(query="daily goals", limit=5)

            # Generate goals based on memories and current state
            for memory in memories:
                goal = Goal(
                    name=f"Goal_{len(goals)}",
                    description=memory.description,
                    deadline=current_state["current_time"] + timedelta(days=1),
                    priority=self._calculate_priority(memory),
                    progress=0.0,
                    status="pending",
                    subgoals=[],
                )
                goals.append(goal)

            return goals

        except Exception as e:
            logger.error(f"Error generating goals: {e}")
            return []

    def _generate_actions(
        self, persona: Any, goals: List[Goal], maze: Any
    ) -> List[Action]:
        """Generate actions to achieve goals."""
        try:
            actions = []

            # Check for reactions
            retrieved = persona.memory.retrieve_relevant_memories(
                query="interaction with other personas", limit=5, recency_weight=0.7
            )

            if self._should_react(persona, retrieved, maze.personas):
                reaction = self._create_react(persona, retrieved, maze.personas, maze)
                if reaction:
                    actions.append(reaction)

            # Generate actions for goals
            for goal in goals:
                # Generate base action
                base_action = Action(
                    name=f"Action_{len(actions)}",
                    description=goal.description,
                    start_time=datetime.now(),
                    end_time=datetime.now() + self.time_slot_duration,
                    location=self._get_action_location(maze),
                    priority=goal.priority,
                    dependencies=[],
                    status="pending",
                )

                # Decompose if needed
                if self._should_decompose(
                    goal.description, self.time_slot_duration.total_seconds() / 60
                ):
                    decomposed_tasks = self._decompose_task(
                        goal.description,
                        int(self.time_slot_duration.total_seconds() / 60),
                    )

                    # Create actions for each subtask
                    for subtask in decomposed_tasks:
                        action = Action(
                            name=f"Action_{len(actions)}",
                            description=subtask["description"],
                            start_time=base_action.start_time,
                            end_time=base_action.start_time
                            + timedelta(minutes=subtask["duration"]),
                            location=base_action.location,
                            priority=base_action.priority,
                            dependencies=[base_action.name],
                            status="pending",
                        )
                        actions.append(action)
                else:
                    actions.append(base_action)

            return actions

        except Exception as e:
            logger.error(f"Error generating actions: {e}")
            return []

    def _compress_schedule(self, schedule: List[Action]) -> List[Action]:
        """Compress schedule by combining consecutive similar actions.

        Args:
            schedule: List of actions to compress

        Returns:
            Compressed schedule
        """
        if not schedule:
            return []

        compressed = []
        current = schedule[0]

        for next_action in schedule[1:]:
            # Check if actions can be combined
            if (
                current.location == next_action.location
                and current.description == next_action.description
                and current.end_time == next_action.start_time
            ):
                # Combine actions
                current.end_time = next_action.end_time
            else:
                compressed.append(current)
                current = next_action

        compressed.append(current)
        return compressed

    def _optimize_schedule(
        self, actions: List[Action], start_time: datetime
    ) -> Dict[str, Any]:
        """Optimize the schedule for the day."""
        try:
            schedule = {}
            current_time = start_time

            # Sort actions by priority
            sorted_actions = sorted(actions, key=lambda x: x.priority, reverse=True)

            # Allocate time slots
            for action in sorted_actions:
                schedule[current_time] = action
                current_time += self.time_slot_duration

            # Compress schedule
            compressed_schedule = self._compress_schedule(sorted_actions)

            # Ensure no time gaps
            for i in range(len(compressed_schedule) - 1):
                if (
                    compressed_schedule[i].end_time
                    < compressed_schedule[i + 1].start_time
                ):
                    # Add filler action
                    filler = Action(
                        name=f"Filler_{i}",
                        description="Free time",
                        start_time=compressed_schedule[i].end_time,
                        end_time=compressed_schedule[i + 1].start_time,
                        location=compressed_schedule[i].location,
                        priority=0.1,
                        dependencies=[],
                        status="pending",
                    )
                    compressed_schedule.insert(i + 1, filler)

            return compressed_schedule

        except Exception as e:
            logger.error(f"Error optimizing schedule: {e}")
            return {}

    def _calculate_priority(self, memory: Any) -> float:
        """Calculate priority for a goal or action."""
        try:
            # Base priority from memory importance
            base_priority = memory.importance

            # Add random variation
            variation = random.uniform(0.0, 0.1)

            return min(1.0, base_priority + variation)

        except Exception as e:
            logger.error(f"Error calculating priority: {e}")
            return 0.5

    def _get_action_location(self, maze: Any) -> Dict[str, Any]:
        """Get a suitable location for an action."""
        try:
            # Get available locations from maze
            locations = maze.get_available_locations()

            # Select a random location
            return random.choice(locations)

        except Exception as e:
            logger.error(f"Error getting action location: {e}")
            return {}

    def _decompose_task(self, task: str, duration: int) -> List[Dict[str, Any]]:
        """Decompose a task into smaller subtasks.

        Args:
            task: Description of the task to decompose
            duration: Total duration of the task in minutes

        Returns:
            List of decomposed tasks with their durations
        """
        try:
            # Get task decomposition from LLM
            decomposition = self.persona.llm.generate_task_decomposition(
                task=task, duration=duration
            )

            # Validate decomposition
            total_duration = sum(subtask["duration"] for subtask in decomposition)
            if total_duration != duration:
                # Adjust durations proportionally if they don't sum to total
                for subtask in decomposition:
                    subtask["duration"] = int(
                        subtask["duration"] * duration / total_duration
                    )

            return decomposition
        except Exception as e:
            logger.error(f"Error decomposing task: {e}")
            # Return single task as fallback
            return [{"description": task, "duration": duration}]

    def _should_decompose(self, task: str, duration: int) -> bool:
        """Determine if a task should be decomposed.

        Args:
            task: Description of the task
            duration: Duration of the task in minutes

        Returns:
            True if task should be decomposed, False otherwise
        """
        # Don't decompose sleeping tasks
        if "sleep" in task.lower() or "bed" in task.lower():
            return False

        # Decompose tasks longer than 1 hour
        if duration > 60:
            return True

        return False

    def _should_react(
        self, persona: Any, retrieved: List[Dict[str, Any]], personas: List[Any]
    ) -> bool:
        """Determine if persona should react to an event.

        Args:
            persona: The persona to check
            retrieved: List of retrieved memories
            personas: List of other personas

        Returns:
            True if persona should react, False otherwise
        """
        try:
            # Check if there are any relevant memories
            if not retrieved:
                return False

            # Get the most recent memory
            memory = retrieved[0]

            # Check if memory is about another persona
            if "persona" not in memory.get("description", "").lower():
                return False

            # Check if the other persona is nearby
            other_persona = next(
                (p for p in personas if p.name in memory["description"]), None
            )
            if not other_persona:
                return False

            # Check if personas are in the same location
            if (
                persona.scratch.current_location
                != other_persona.scratch.current_location
            ):
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking reaction: {e}")
            return False

    def _create_react(
        self,
        persona: Any,
        retrieved: List[Dict[str, Any]],
        personas: List[Any],
        maze: Any,
    ) -> Action:
        """Create a reaction action.

        Args:
            persona: The persona reacting
            retrieved: List of retrieved memories
            personas: List of other personas
            maze: The maze object

        Returns:
            Reaction action
        """
        try:
            # Get the most recent memory
            memory = retrieved[0]

            # Get the other persona
            other_persona = next(
                (p for p in personas if p.name in memory["description"]), None
            )

            # Generate reaction description
            reaction = self.persona.llm.generate_reaction(
                persona=persona, other_persona=other_persona, memory=memory
            )

            # Create reaction action
            action = Action(
                name=f"React_{len(persona.scratch.actions)}",
                description=reaction,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(minutes=5),
                location=persona.scratch.current_location,
                priority=0.8,
                dependencies=[],
                status="pending",
            )

            return action

        except Exception as e:
            logger.error(f"Error creating reaction: {e}")
            return None


class Plan:
    """Main planning module for generative agents."""

    def __init__(self):
        self.plan_generator = PlanGenerator()

    def generate_daily_plan(
        self, persona: Any, maze: Any, start_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate a daily plan for the agent."""
        return self.plan_generator.generate_daily_plan(
            persona=persona, maze=maze, start_time=start_time
        )

    def generate_action_sequence(self, persona: Any, maze: Any) -> List[Dict[str, Any]]:
        """Generate a sequence of actions for the agent."""
        try:
            # Get current plan
            plan = self.generate_daily_plan(persona, maze)

            # Extract action sequence
            actions = plan.get("actions", [])

            # Sort by start time
            sorted_actions = sorted(actions, key=lambda x: x["start_time"])

            return sorted_actions

        except Exception as e:
            logger.error(f"Error generating action sequence: {e}")
            return []

    def update_plan(
        self, persona: Any, maze: Any, changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update the current plan with changes."""
        try:
            # Get current plan
            plan = self.generate_daily_plan(persona, maze)

            # Apply changes
            for key, value in changes.items():
                if key in plan:
                    plan[key] = value

            return plan

        except Exception as e:
            logger.error(f"Error updating plan: {e}")
            return {}

    def generate(self, persona: Any, maze: Any) -> Dict[str, Any]:
        """Generate a plan for the persona."""
        try:
            # Generate goals
            goals = self.plan_generator._generate_goals(
                persona, persona.scratch.get_current_state()
            )

            # Generate actions
            actions = self.plan_generator._generate_actions(persona, goals, maze)

            # Optimize schedule
            optimized_actions = self.plan_generator._optimize_schedule(
                actions, datetime.now()
            )

            # Create plan
            plan = {
                "goals": [goal.to_dict() for goal in goals],
                "actions": [action.to_dict() for action in optimized_actions],
                "created": datetime.now().isoformat(),
                "status": "active",
            }

            return plan

        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            return {}
