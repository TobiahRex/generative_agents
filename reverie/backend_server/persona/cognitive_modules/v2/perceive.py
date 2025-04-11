"""
Perceive module for generative agents.

This module handles perception and awareness for generative agents, including:
- Event detection and processing
- Spatial awareness
- Memory integration
- State management
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
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
class Event:
    """Represents a perceived event in the environment."""

    subject: str
    predicate: str
    object: str
    description: str
    importance: float
    created: datetime
    expiration: datetime
    subject_location: Dict[str, Any]
    object_location: Dict[str, Any]

    def to_triple(self) -> Tuple[str, str, str]:
        """Convert event to a triple representation."""
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "importance": self.importance,
            "created": self.created,
            "expiration": self.expiration,
            "subject_location": self.subject_location,
            "object_location": self.object_location,
        }


class EventProcessor:
    """Handles event processing and importance scoring."""

    def __init__(self):
        self.importance_weights = {
            "chat": 1.0,
            "chat_idle": 0.3,
            "move": 0.5,
            "move_idle": 0.2,
            "action": 0.8,
            "action_idle": 0.3,
        }

    def process_event(self, event: Dict[str, Any]) -> Event:
        """Process a raw event into an Event object."""
        try:
            # Extract event components
            subject = event.get("subject", "")
            predicate = event.get("predicate", "")
            obj = event.get("object", "")
            description = event.get("description", "")

            # Calculate importance
            importance = self._calculate_importance(event)

            # Set timestamps
            created = datetime.now()
            expiration = created.replace(hour=23, minute=59, second=59)

            # Get locations
            subject_location = event.get("subject_location", {})
            object_location = event.get("object_location", {})

            return Event(
                subject=subject,
                predicate=predicate,
                object=obj,
                description=description,
                importance=importance,
                created=created,
                expiration=expiration,
                subject_location=subject_location,
                object_location=object_location,
            )
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            raise

    def _calculate_importance(self, event: Dict[str, Any]) -> float:
        """Calculate the importance of an event."""
        try:
            # Base importance from weights
            event_type = event.get("type", "")
            base_importance = self.importance_weights.get(event_type, 0.5)

            # Add random variation
            variation = random.uniform(0.0, 0.1)

            return min(1.0, base_importance + variation)
        except Exception as e:
            logger.error(f"Error calculating importance: {e}")
            return 0.5


class Perceive:
    """Main perception module for generative agents."""

    def __init__(self):
        self.event_processor = EventProcessor()
        self.retention_threshold = 0.5

    def process_events(
        self, persona: Any, maze: Any, events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process and store events in the agent's memory.

        Args:
            persona: The agent's persona
            maze: The environment maze
            events: List of events to process

        Returns:
            List of processed events
        """
        try:
            processed_events = []

            for event in events:
                # Process the event
                processed_event = self.event_processor.process_event(event)

                # Store in memory if important enough
                if processed_event.importance >= self.retention_threshold:
                    self._store_in_memory(persona, processed_event)

                processed_events.append(processed_event.to_dict())

            return processed_events

        except Exception as e:
            logger.error(f"Error processing events: {e}")
            return []

    def _store_in_memory(self, persona: Any, event: Event) -> None:
        """Store an event in the agent's memory."""
        try:
            # Store in associative memory
            persona.a_mem.add_memory(
                event.description, event.importance, event.created, event.expiration
            )

            # Store in spatial memory if locations are available
            if event.subject_location and event.object_location:
                persona.s_mem.add_memory(
                    event.subject,
                    event.subject_location,
                    event.object,
                    event.object_location,
                    event.created,
                )

        except Exception as e:
            logger.error(f"Error storing event in memory: {e}")

    def update_state(self, persona: Any, maze: Any) -> None:
        """Update the agent's current state based on perception."""
        try:
            # Update current tile
            curr_tile = maze.access_tile(persona.scratch.curr_tile)
            persona.scratch.curr_tile = curr_tile

            # Update current time
            curr_time = datetime.now()
            persona.scratch.curr_time = curr_time

            # Update current action
            curr_action = persona.scratch.act_address.split(":")[0]
            persona.scratch.curr_action = curr_action

        except Exception as e:
            logger.error(f"Error updating state: {e}")

    def get_current_state(self, persona: Any) -> Dict[str, Any]:
        """Get the agent's current state."""
        try:
            return {
                "current_tile": persona.scratch.curr_tile,
                "current_time": persona.scratch.curr_time,
                "current_action": persona.scratch.curr_action,
                "daily_plan": persona.scratch.daily_plan,
                "daily_req": persona.scratch.daily_req,
                "f_daily_schedule": persona.scratch.f_daily_schedule,
            }
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            return {}
