"""
Execute module for generative agents.

This module handles the execution of actions and movements for generative agents, including:
- Pathfinding and movement
- Action execution
- Interaction management
- Memory integration
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum
import numpy as np

from reverie.backend_server.persona.cognitive_modules.v2.retrieve import (
    Retrieve,
    Triplet,
)
from reverie.backend_server.persona.memory_structures.v2.associative_memory import (
    AssociativeMemory,
)

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions that can be executed."""

    MOVEMENT = "movement"
    INTERACTION = "interaction"
    OBJECT_MANIPULATION = "object_manipulation"
    COMMUNICATION = "communication"


@dataclass
class ActionAddress:
    """Represents an action address in the world."""

    world: str
    sector: str
    arena: str
    game_object: Optional[str] = None

    @classmethod
    def from_string(cls, address: str) -> "ActionAddress":
        """Parse an action address string into components."""
        parts = address.split(":")
        return cls(
            world=parts[0],
            sector=parts[1],
            arena=parts[2],
            game_object=parts[3] if len(parts) > 3 else None,
        )


@dataclass
class Path:
    """Represents a movement path."""

    start: Tuple[int, int]
    end: Tuple[int, int]
    waypoints: List[Tuple[int, int]]
    created: datetime = datetime.now()

    def optimize(self) -> None:
        """Optimize the path by removing unnecessary waypoints."""
        if len(self.waypoints) < 3:
            return

        optimized = [self.waypoints[0]]
        for i in range(1, len(self.waypoints) - 1):
            if not self._is_collinear(
                optimized[-1], self.waypoints[i], self.waypoints[i + 1]
            ):
                optimized.append(self.waypoints[i])
        optimized.append(self.waypoints[-1])
        self.waypoints = optimized

    def _is_collinear(
        self, p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]
    ) -> bool:
        """Check if three points are collinear."""
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])


@dataclass
class Action:
    """Represents an action to be executed."""

    address: ActionAddress
    description: str
    duration: int
    type: ActionType
    created: datetime = datetime.now()
    expiration: datetime = None
    triplet: Optional[Triplet] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.expiration is None:
            self.expiration = self.created + datetime.timedelta(hours=1)

    def validate(self) -> bool:
        """Validate the action."""
        return (
            self.address is not None
            and self.description
            and self.duration > 0
            and self.type in ActionType
        )


@dataclass
class ExecutionResult:
    """Represents the result of an action execution."""

    success: bool
    error: Optional[str] = None
    description: Optional[str] = None
    path: Optional[Path] = None
    duration: Optional[int] = None


class PathFinder:
    """Handles pathfinding operations."""

    def __init__(self, maze: Any):
        self.maze = maze
        self.cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Path] = {}

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[Path]:
        """Find a path from start to end using A* algorithm."""
        cache_key = (start, end)
        if cache_key in self.cache:
            return self.cache[cache_key]

        path = self._a_star(start, end)
        if path:
            self.cache[cache_key] = path
        return path

    def _a_star(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[Path]:
        """A* pathfinding algorithm implementation."""
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float("inf")))
            if current == end:
                return self._reconstruct_path(came_from, current)

            open_set.remove(current)
            for neighbor in self._get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(
                        neighbor, end
                    )
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return None

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic function for A* (Manhattan distance)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_position(new_pos):
                neighbors.append(new_pos)
        return neighbors

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid in the maze."""
        return (
            0 <= pos[0] < self.maze.width
            and 0 <= pos[1] < self.maze.height
            and not self.maze.is_collision(pos)
        )

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> Path:
        """Reconstruct the path from the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return Path(start=path[0], end=path[-1], waypoints=path)


class Execute:
    """Main execution module for generative agents."""

    def __init__(self, persona: Any, maze: Any, personas: Dict[str, Any]):
        self.persona = persona
        self.maze = maze
        self.personas = personas
        self.path_finder = PathFinder(maze)
        self.retrieve = Retrieve(persona)

    def execute_action(self, plan: str) -> ExecutionResult:
        """Execute an action plan."""
        try:
            # Parse the action address
            address = ActionAddress.from_string(plan)

            # Create action object
            action = self._create_action(address)
            if not action.validate():
                return ExecutionResult(success=False, error="Invalid action")

            # Find and validate path
            path = self._find_path_for_action(action)
            if not path:
                return ExecutionResult(success=False, error="No valid path found")

            # Execute movement
            if not self._execute_movement(path):
                return ExecutionResult(success=False, error="Movement failed")

            # Execute action
            if not self._execute_action(action):
                return ExecutionResult(success=False, error="Action execution failed")

            # Update memory
            self._update_memory(action, path)

            return ExecutionResult(
                success=True,
                description=action.description,
                path=path,
                duration=action.duration,
            )

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return ExecutionResult(success=False, error=str(e))

    def _create_action(self, address: ActionAddress) -> Action:
        """Create an action object from an address."""
        # Get action details from the persona's scratch
        description = self.persona.scratch.act_description
        duration = self.persona.scratch.act_duration
        action_type = self._determine_action_type(address, description)

        # Generate triplet and embedding
        triplet = self._generate_triplet(description)
        embedding = self._get_embedding(description)

        return Action(
            address=address,
            description=description,
            duration=duration,
            type=action_type,
            triplet=triplet,
            embedding=embedding,
        )

    def _determine_action_type(
        self, address: ActionAddress, description: str
    ) -> ActionType:
        """Determine the type of action based on address and description."""
        if "<persona>" in description:
            return ActionType.INTERACTION
        elif "<random>" in description:
            return ActionType.MOVEMENT
        elif address.game_object:
            return ActionType.OBJECT_MANIPULATION
        else:
            return ActionType.MOVEMENT

    def _find_path_for_action(self, action: Action) -> Optional[Path]:
        """Find a path for the action."""
        if action.type == ActionType.INTERACTION:
            target_persona = self.personas[
                action.description.split("<persona>")[-1].strip()
            ]
            target_tile = target_persona.scratch.curr_tile
            return self.path_finder.find_path(
                self.persona.scratch.curr_tile, target_tile
            )
        else:
            target_tiles = self.maze.address_tiles.get(str(action.address), [])
            if not target_tiles:
                return None

            # Find the closest valid target tile
            closest_path = None
            min_length = float("inf")
            for tile in target_tiles:
                path = self.path_finder.find_path(self.persona.scratch.curr_tile, tile)
                if path and len(path.waypoints) < min_length:
                    closest_path = path
                    min_length = len(path.waypoints)

            return closest_path

    def _execute_movement(self, path: Path) -> bool:
        """Execute movement along a path."""
        try:
            for waypoint in path.waypoints[1:]:  # Skip start position
                self.persona.scratch.curr_tile = waypoint
                # Add any movement-related memory updates here
            return True
        except Exception as e:
            logger.error(f"Error executing movement: {e}")
            return False

    def _execute_action(self, action: Action) -> bool:
        """Execute the actual action."""
        try:
            if action.type == ActionType.INTERACTION:
                return self._execute_interaction(action)
            elif action.type == ActionType.OBJECT_MANIPULATION:
                return self._execute_object_manipulation(action)
            else:
                return True  # Movement actions are handled separately
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return False

    def _execute_interaction(self, action: Action) -> bool:
        """Execute an interaction action."""
        target_name = action.description.split("<persona>")[-1].strip()
        target_persona = self.personas.get(target_name)
        if not target_persona:
            return False

        # Add interaction logic here
        return True

    def _execute_object_manipulation(self, action: Action) -> bool:
        """Execute an object manipulation action."""
        # Add object manipulation logic here
        return True

    def _update_memory(self, action: Action, path: Path) -> None:
        """Update memory with action and path information."""
        try:
            # Add action to memory
            self.persona.a_mem.add_action(
                created=action.created,
                expiration=action.expiration,
                address=str(action.address),
                description=action.description,
                duration=action.duration,
                triplet=action.triplet,
                embedding=action.embedding,
            )

            # Add path to memory
            self.persona.a_mem.add_path(
                created=path.created,
                start=path.start,
                end=path.end,
                waypoints=path.waypoints,
            )
        except Exception as e:
            logger.error(f"Error updating memory: {e}")

    def _generate_triplet(self, description: str) -> Optional[Triplet]:
        """Generate a triplet from action description."""
        try:
            # Use the persona's LLM to generate the triplet
            s, p, o = self.persona.llm.generate_action_event_triple(
                description, self.persona
            )
            return Triplet(subject=s, predicate=p, object=o)
        except Exception as e:
            logger.error(f"Error generating triplet: {e}")
            return None

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        try:
            embedding = self.persona.llm.get_embedding(
                text=text, model="text-embedding-ada-002"
            )
            # Normalize the embedding
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0.0] * 1536  # Return zero vector of correct dimension
