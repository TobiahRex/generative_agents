"""
Reflect module for generative agents.

This module handles reflection and insight generation for generative agents, including:
- Focal point generation
- Insight and evidence generation
- Memory analysis and pattern recognition
- Reflection triggering and management
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from numpy import dot
from numpy.linalg import norm

from reverie.backend_server.persona.cognitive_modules.v2.retrieve import (
    Retrieve,
    Triplet,
)
from reverie.backend_server.persona.cognitive_modules.v2.associative_memory import (
    AssociativeMemory,
)

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """Represents an insight generated during reflection."""

    content: str
    evidence: List[str]  # List of node_ids that support this insight
    importance: float
    created: datetime
    expiration: datetime
    triplet: Optional[Triplet] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary representation."""
        return {
            "content": self.content,
            "evidence": self.evidence,
            "importance": self.importance,
            "created": self.created,
            "expiration": self.expiration,
            "triplet": self.triplet.to_dict() if self.triplet else None,
        }


@dataclass
class FocalPoint:
    """Represents a focal point for reflection."""

    content: str
    importance: float
    created: datetime
    related_memories: List[str]  # List of node_ids
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert focal point to dictionary representation."""
        return {
            "content": self.content,
            "importance": self.importance,
            "created": self.created,
            "related_memories": self.related_memories,
        }


class FocalPointGenerator:
    """Handles generation of focal points for reflection."""

    def __init__(self, persona: Any):
        self.persona = persona
        self.retrieve = Retrieve(persona)
        self.max_focal_points = 3
        self.min_importance = 0.3

    def generate_focal_points(self, n: Optional[int] = None) -> List[FocalPoint]:
        """Generate focal points from recent memories."""
        try:
            n = n or self.max_focal_points

            # Get recent memories
            recent_memories = self._get_recent_memories()

            # Generate focal points using GPT
            focal_points = self._generate_focal_points_gpt(recent_memories, n)

            # Process and validate focal points
            processed_points = []
            for point in focal_points:
                if self._validate_focal_point(point):
                    processed_points.append(point)

            return processed_points[:n]
        except Exception as e:
            logger.error(f"Error generating focal points: {e}")
            return []

    def _get_recent_memories(self) -> List[Dict[str, Any]]:
        """Get recent memories for focal point generation."""
        try:
            # Get recent events and thoughts
            memories = []

            # Add recent events
            for event in self.persona.a_mem.seq_event[
                -self.persona.scratch.importance_ele_n :
            ]:
                if "idle" not in event.embedding_key:
                    memories.append(
                        {
                            "content": event.embedding_key,
                            "importance": event.poignancy,
                            "created": event.created,
                            "node_id": event.node_id,
                        }
                    )

            # Add recent thoughts
            for thought in self.persona.a_mem.seq_thought[
                -self.persona.scratch.importance_ele_n :
            ]:
                memories.append(
                    {
                        "content": thought.embedding_key,
                        "importance": thought.poignancy,
                        "created": thought.created,
                        "node_id": thought.node_id,
                    }
                )

            return sorted(memories, key=lambda x: x["created"], reverse=True)
        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return []

    def _generate_focal_points_gpt(
        self, memories: List[Dict[str, Any]], n: int
    ) -> List[FocalPoint]:
        """Generate focal points using GPT."""
        try:
            # Format memories for GPT
            statements = "\n".join([m["content"] for m in memories])

            # Call GPT to generate focal points
            focal_points = self.persona.llm.generate_focal_points(
                statements=statements, n=n, persona=self.persona
            )

            # Process GPT output into FocalPoint objects
            processed_points = []
            for point in focal_points:
                # Find related memories
                related_memories = self._find_related_memories(point, memories)

                # Create FocalPoint object
                processed_points.append(
                    FocalPoint(
                        content=point,
                        importance=self._calculate_importance(point, related_memories),
                        created=datetime.now(),
                        related_memories=[m["node_id"] for m in related_memories],
                        embedding=self._get_embedding(point),
                    )
                )

            return processed_points
        except Exception as e:
            logger.error(f"Error generating focal points with GPT: {e}")
            return []

    def _validate_focal_point(self, point: FocalPoint) -> bool:
        """Validate a focal point."""
        return (
            point.importance >= self.min_importance and len(point.related_memories) > 0
        )

    def _find_related_memories(
        self, point: str, memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find memories related to a focal point.

        Args:
            point: The focal point to find related memories for
            memories: List of memories to search through

        Returns:
            List of related memories with their metadata
        """
        try:
            # Use the retrieve module to find related memories
            results = self.retrieve.retrieve_memories(
                query=point,
                limit=5,  # Limit to top 5 most relevant memories
                time_range=(
                    datetime.now() - timedelta(days=7),
                    datetime.now(),
                ),  # Last 7 days
            )

            # Map results back to original memories
            related_memories = []
            for result in results:
                for memory in memories:
                    if memory["node_id"] == result["node_id"]:
                        # Add relevance score to memory metadata
                        memory["relevance_score"] = result["relevance_score"]
                        related_memories.append(memory)
                        break

            # Sort by relevance score
            related_memories.sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )

            return related_memories

        except Exception as e:
            logger.error(f"Error finding related memories: {e}")
            return []

    def _calculate_importance(
        self, point: str, related_memories: List[Dict[str, Any]]
    ) -> float:
        """Calculate importance score for a focal point.

        The importance score is based on:
        1. Average importance of related memories
        2. Number of related memories
        3. Recency of related memories
        4. Relevance scores of related memories

        Args:
            point: The focal point to score
            related_memories: List of related memories with metadata

        Returns:
            Importance score between 0 and 1
        """
        try:
            if not related_memories:
                return 0.0

            # Calculate base importance from memory importance scores
            total_importance = sum(m["importance"] for m in related_memories)
            avg_importance = total_importance / len(related_memories)

            # Calculate recency factor (more recent = higher score)
            now = datetime.now()
            recency_scores = [
                1.0
                / (
                    1.0 + (now - m["created"]).total_seconds() / 86400
                )  # Decay over days
                for m in related_memories
            ]
            recency_factor = sum(recency_scores) / len(recency_scores)

            # Calculate relevance factor
            relevance_scores = [m.get("relevance_score", 0) for m in related_memories]
            relevance_factor = sum(relevance_scores) / len(relevance_scores)

            # Calculate quantity factor (more memories = higher score)
            quantity_factor = min(len(related_memories) / 5.0, 1.0)  # Cap at 5 memories

            # Combine factors with weights
            importance = (
                0.4 * avg_importance  # Base importance
                + 0.2 * recency_factor  # Recency
                + 0.2 * relevance_factor  # Relevance
                + 0.2 * quantity_factor  # Quantity
            )

            return min(max(importance, 0.0), 1.0)  # Ensure between 0 and 1

        except Exception as e:
            logger.error(f"Error calculating importance: {e}")
            return 0.0

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a focal point.

        Args:
            text: The focal point text to embed

        Returns:
            List of floats representing the embedding
        """
        try:
            # Use the persona's embedding generator
            embedding = self.persona.llm.get_embedding(
                text=text, model="text-embedding-ada-002"  # OpenAI's embedding model
            )

            # Normalize the embedding
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 1536  # Return zero vector of correct dimension


class InsightGenerator:
    """Handles generation of insights and evidence."""

    def __init__(self, persona: Any):
        self.persona = persona
        self.retrieve = Retrieve(persona)
        self.max_insights = 5
        self.min_importance = 0.3

    def generate_insights(
        self, nodes: List[Any], n: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """Generate insights and evidence from nodes."""
        try:
            n = n or self.max_insights

            # Format nodes for GPT
            statements = self._format_nodes(nodes)

            # Generate insights using GPT
            insights = self._generate_insights_gpt(statements, n)

            # Process and validate insights
            processed_insights = {}
            for insight, evidence in insights.items():
                if self._validate_insight(insight, evidence):
                    processed_insights[insight] = evidence

            return processed_insights
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {}

    def _format_nodes(self, nodes: List[Any]) -> str:
        """Format nodes for GPT input."""
        try:
            statements = ""
            for count, node in enumerate(nodes):
                statements += f"{str(count)}. {node.embedding_key}\n"
            return statements
        except Exception as e:
            logger.error(f"Error formatting nodes: {e}")
            return ""

    def _generate_insights_gpt(self, statements: str, n: int) -> Dict[str, List[str]]:
        """Generate insights using GPT."""
        try:
            # Call GPT to generate insights
            insights = self.persona.llm.generate_insights(
                statements=statements, n=n, persona=self.persona
            )

            # Process GPT output
            processed_insights = {}
            for thought, evidence_indices in insights.items():
                # Convert evidence indices to node_ids
                evidence_node_ids = [nodes[i].node_id for i in evidence_indices]
                processed_insights[thought] = evidence_node_ids

            return processed_insights
        except Exception as e:
            logger.error(f"Error generating insights with GPT: {e}")
            return {}

    def _validate_insight(self, insight: str, evidence: List[str]) -> bool:
        """Validate an insight."""
        return (
            len(evidence) > 0
            and self._calculate_insight_importance(insight, evidence)
            >= self.min_importance
        )

    def _calculate_insight_importance(self, insight: str, evidence: List[str]) -> float:
        """Calculate importance of an insight."""
        try:
            # Get the nodes for the evidence
            nodes = []
            for node_id in evidence:
                node = self.persona.a_mem.id_to_node.get(node_id)
                if node:
                    nodes.append(node)

            if not nodes:
                return 0.0

            # Average importance of evidence nodes
            total_importance = sum(node.poignancy for node in nodes)
            return total_importance / len(nodes)
        except Exception as e:
            logger.error(f"Error calculating insight importance: {e}")
            return 0.0


class Reflect:
    """Main reflection module for generative agents."""

    def __init__(self, persona: Any):
        self.persona = persona
        self.focal_point_generator = FocalPointGenerator(persona)
        self.insight_generator = InsightGenerator(persona)
        self.retrieve = Retrieve(persona)

    def reflect(self) -> None:
        """Run the reflection process."""
        try:
            # Check if reflection should be triggered
            if not self._should_reflect():
                return

            # Generate focal points
            focal_points = self.focal_point_generator.generate_focal_points()

            # Process each focal point
            for point in focal_points:
                # Retrieve relevant memories
                memories = self.retrieve.retrieve_memories(
                    query=point.content, limit=10
                )

                # Generate insights
                insights = self.insight_generator.generate_insights(memories)

                # Store insights
                self._store_insights(insights, point)

            # Reset reflection trigger
            self._reset_reflection_trigger()
        except Exception as e:
            logger.error(f"Error in reflection process: {e}")

    def _should_reflect(self) -> bool:
        """Check if reflection should be triggered."""
        try:
            return self.persona.scratch.importance_trigger_curr <= 0 and bool(
                self.persona.a_mem.seq_event + self.persona.a_mem.seq_thought
            )
        except Exception as e:
            logger.error(f"Error checking reflection trigger: {e}")
            return False

    def _reset_reflection_trigger(self) -> None:
        """Reset the reflection trigger counter."""
        try:
            self.persona.scratch.importance_trigger_curr = (
                self.persona.scratch.importance_trigger_max
            )
            self.persona.scratch.importance_ele_n = 0
        except Exception as e:
            logger.error(f"Error resetting reflection trigger: {e}")

    def _store_insights(
        self, insights: Dict[str, List[str]], focal_point: FocalPoint
    ) -> None:
        """Store generated insights in memory."""
        try:
            for thought, evidence in insights.items():
                # Generate triplet
                triplet = self._generate_triplet(thought)

                # Calculate importance
                importance = self._calculate_thought_importance(thought, evidence)

                # Create insight object
                insight = Insight(
                    content=thought,
                    evidence=evidence,
                    importance=importance,
                    created=datetime.now(),
                    expiration=datetime.now() + timedelta(days=30),
                    triplet=triplet,
                    embedding=self._get_embedding(thought),
                )

                # Add to associative memory
                self.persona.a_mem.add_thought(
                    created=insight.created,
                    expiration=insight.expiration,
                    subject=triplet.subject if triplet else "",
                    predicate=triplet.predicate if triplet else "",
                    object=triplet.object if triplet else "",
                    content=insight.content,
                    keywords=(
                        set([triplet.subject, triplet.predicate, triplet.object])
                        if triplet
                        else set()
                    ),
                    poignancy=insight.importance,
                    embedding_pair=(insight.content, insight.embedding),
                    evidence=insight.evidence,
                )
        except Exception as e:
            logger.error(f"Error storing insights: {e}")

    def _generate_triplet(self, thought: str) -> Optional[Triplet]:
        """Generate SPO triplet from thought."""
        try:
            # Call GPT to generate triplet
            s, p, o = self.persona.llm.generate_action_event_triple(
                thought, self.persona
            )
            return Triplet(subject=s, predicate=p, object=o)
        except Exception as e:
            logger.error(f"Error generating triplet: {e}")
            return None

    def _calculate_thought_importance(self, thought: str, evidence: List[str]) -> float:
        """Calculate importance of a thought."""
        try:
            if "is idle" in thought:
                return 1.0

            # Call GPT to calculate importance
            return self.persona.llm.generate_poignancy_score(
                event_type="thought", description=thought, persona=self.persona
            )
        except Exception as e:
            logger.error(f"Error calculating thought importance: {e}")
            return 0.0

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        # This should use the same embedding method as the retrieve module
        return [0.0] * 1536  # Placeholder
