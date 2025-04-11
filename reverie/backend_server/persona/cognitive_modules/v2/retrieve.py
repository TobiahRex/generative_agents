"""
Retrieve module for generative agents.

This module handles memory retrieval for generative agents, including:
- Query processing
- Memory search
- Result filtering
- Result sorting
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm

from reverie.backend_server.persona.memory_structures.v2.associative_memory import (
    AssociativeMemory,
)
from reverie.backend_server.persona.memory_structures.v2.spatial_memory import (
    SpatialMemory,
)
from reverie.backend_server.persona.memory_structures.v2.scratch import Scratch

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """Represents a subject-predicate-object triplet."""

    subject: str
    predicate: str
    object: str

    def to_dict(self) -> Dict[str, str]:
        """Convert triplet to dictionary representation."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
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


@dataclass
class SearchResult:
    """Represents a search result from memory."""

    content: str
    importance: float
    created: datetime
    memory_type: str
    metadata: Dict[str, Any]
    relevance_score: float = 0.0
    recency_score: float = 0.0
    embedding: Optional[List[float]] = None
    node_id: Optional[str] = None
    triplet: Optional[Triplet] = None  # Added SPO triplet

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary representation."""
        return {
            "content": self.content,
            "importance": self.importance,
            "created": self.created,
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "recency_score": self.recency_score,
            "node_id": self.node_id,
            "triplet": self.triplet.to_dict() if self.triplet else None,
        }


class QueryProcessor:
    """Handles query processing and search term generation."""

    def __init__(self):
        self.keyword_pattern = re.compile(r"\b\w+\b")
        self.stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to"}

    def process_query(
        self, query: str
    ) -> Tuple[List[str], List[float], Optional[Triplet]]:
        """Process a query into search terms and get its embedding."""
        try:
            # Parse the query
            parsed_query = self._parse_query(query)

            # Extract keywords
            keywords = self._extract_keywords(parsed_query)

            # Generate search terms
            search_terms = self._generate_search_terms(keywords)

            # Get query embedding
            query_embedding = self._get_embedding(query)

            # Extract SPO triplet if possible
            triplet = self._extract_triplet(parsed_query)

            return search_terms, query_embedding, triplet
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return [], [], None

    def _extract_triplet(self, query: str) -> Optional[Triplet]:
        """Extract subject-predicate-object triplet from query."""
        try:
            # Simple pattern matching for now
            # This could be enhanced with NLP techniques
            words = query.split()
            if len(words) >= 3:
                # Try to find a verb (predicate)
                for i in range(1, len(words) - 1):
                    if self._is_verb(words[i]):
                        return Triplet(
                            subject=" ".join(words[:i]),
                            predicate=words[i],
                            object=" ".join(words[i + 1 :]),
                        )
            return None
        except Exception as e:
            logger.error(f"Error extracting triplet: {e}")
            return None

    def _is_verb(self, word: str) -> bool:
        """Simple check if a word might be a verb."""
        # This is a very basic implementation
        # Could be enhanced with a proper verb dictionary or NLP
        common_verbs = {
            "is",
            "was",
            "are",
            "were",
            "has",
            "have",
            "had",
            "do",
            "does",
            "did",
        }
        return word.lower() in common_verbs

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the same method as V1."""
        # This should use the same embedding method as V1
        # For now, returning a placeholder
        return [0.0] * 1536  # Typical embedding dimension

    def _parse_query(self, query: str) -> str:
        """Parse the query into a standardized format."""
        try:
            # Convert to lowercase
            query = query.lower()

            # Remove special characters
            query = re.sub(r"[^\w\s]", "", query)

            return query
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return ""

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from the query."""
        try:
            # Find all words
            words = self.keyword_pattern.findall(query)

            # Remove stop words
            keywords = [word for word in words if word not in self.stop_words]

            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    def _generate_search_terms(self, keywords: List[str]) -> List[str]:
        """Generate search terms from keywords."""
        try:
            # Add variations of keywords
            search_terms = []
            for keyword in keywords:
                search_terms.append(keyword)
                search_terms.append(f"{keyword}s")  # Plural form
                search_terms.append(f"{keyword}ing")  # Present participle

            return list(set(search_terms))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error generating search terms: {e}")
            return []


class ResultProcessor:
    """Handles result filtering and sorting."""

    def __init__(self, persona: Any):
        self.persona = persona
        self.min_importance = 0.3
        self.max_results = 100
        self.recency_weight = 1.0
        self.relevance_weight = 1.0
        self.importance_weight = 1.0
        self.recency_decay = 0.99

    def filter_results(
        self,
        results: List[SearchResult],
        search_terms: List[str],
        query_embedding: List[float],
        query_triplet: Optional[Triplet],
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[SearchResult]:
        """Filter search results."""
        try:
            filtered_results = []

            # Calculate recency scores for all results
            self._calculate_recency_scores(results)

            for result in results:
                # Filter by relevance
                if not self._is_relevant(
                    result, search_terms, query_embedding, query_triplet
                ):
                    continue

                # Filter by time
                if not self._is_in_time_range(result, time_range):
                    continue

                # Filter by importance
                if not self._is_important(result):
                    continue

                filtered_results.append(result)

            return filtered_results[: self.max_results]
        except Exception as e:
            logger.error(f"Error filtering results: {e}")
            return []

    def sort_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Sort search results."""
        try:
            # Sort by combined score
            sorted_results = sorted(
                results,
                key=lambda x: (
                    self.recency_weight * x.recency_score
                    + self.relevance_weight * x.relevance_score
                    + self.importance_weight * x.importance
                ),
                reverse=True,
            )

            return sorted_results
        except Exception as e:
            logger.error(f"Error sorting results: {e}")
            return []

    def _calculate_recency_scores(self, results: List[SearchResult]) -> None:
        """Calculate recency scores for all results using V1's decay pattern."""
        try:
            # Sort results by creation time
            sorted_results = sorted(results, key=lambda x: x.created, reverse=True)

            # Calculate recency scores using exponential decay
            for i, result in enumerate(sorted_results):
                result.recency_score = self.recency_decay**i
        except Exception as e:
            logger.error(f"Error calculating recency scores: {e}")

    def _is_relevant(
        self,
        result: SearchResult,
        search_terms: List[str],
        query_embedding: List[float],
        query_triplet: Optional[Triplet],
    ) -> bool:
        """Check if a result is relevant to the search terms."""
        try:
            # Keyword-based relevance
            content = result.content.lower()
            search_terms = [term.lower() for term in search_terms]
            keyword_match = any(term in content for term in search_terms)

            # Vector similarity relevance
            if hasattr(result, "embedding") and result.embedding:
                result.relevance_score = self._cosine_similarity(
                    query_embedding, result.embedding
                )
                vector_match = result.relevance_score > 0.5
            else:
                vector_match = False

            # SPO triplet matching
            triplet_match = False
            if query_triplet and result.triplet:
                triplet_match = (
                    self._match_triplet_component(
                        query_triplet.subject, result.triplet.subject
                    )
                    or self._match_triplet_component(
                        query_triplet.predicate, result.triplet.predicate
                    )
                    or self._match_triplet_component(
                        query_triplet.object, result.triplet.object
                    )
                )

            return keyword_match or vector_match or triplet_match
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return False

    def _match_triplet_component(
        self, query_component: str, result_component: str
    ) -> bool:
        """Check if two triplet components match."""
        try:
            # Simple string matching for now
            # Could be enhanced with semantic similarity
            return (
                query_component.lower() in result_component.lower()
                or result_component.lower() in query_component.lower()
            )
        except Exception as e:
            logger.error(f"Error matching triplet components: {e}")
            return False

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return dot(a, b) / (norm(a) * norm(b))

    def _is_in_time_range(
        self,
        result: SearchResult,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> bool:
        """Check if a result is within the time range."""
        try:
            if time_range is None:
                return True

            start_time, end_time = time_range
            return start_time <= result.created <= end_time
        except Exception as e:
            logger.error(f"Error checking time range: {e}")
            return False

    def _is_important(self, result: SearchResult) -> bool:
        """Check if a result is important enough."""
        try:
            return result.importance >= self.min_importance
        except Exception as e:
            logger.error(f"Error checking importance: {e}")
            return False


class Retrieve:
    """Main memory retrieval module for generative agents."""

    def __init__(self, persona: Any):
        self.persona = persona
        self.query_processor = QueryProcessor()
        self.result_processor = ResultProcessor(persona)

    def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve memories based on a query."""
        try:
            # Process the query
            search_terms, query_embedding, query_triplet = (
                self.query_processor.process_query(query)
            )

            # Search across memory systems
            results = self._search_memories(
                search_terms=search_terms,
                query_embedding=query_embedding,
                query_triplet=query_triplet,
                time_range=time_range,
            )

            # Filter and sort results
            filtered_results = self.result_processor.filter_results(
                results=results,
                search_terms=search_terms,
                query_embedding=query_embedding,
                query_triplet=query_triplet,
                time_range=time_range,
            )
            sorted_results = self.result_processor.sort_results(filtered_results)

            # Update last accessed time for retrieved memories
            self._update_last_accessed(sorted_results)

            # Return the top results
            return [result.to_dict() for result in sorted_results[:limit]]
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    def _search_memories(
        self,
        search_terms: List[str],
        query_embedding: List[float],
        query_triplet: Optional[Triplet],
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> List[SearchResult]:
        """Search across memory systems."""
        try:
            results = []

            # Search associative memory
            a_mem_results = self.persona.a_mem.search(
                terms=search_terms, triplet=query_triplet, time_range=time_range
            )
            results.extend(
                [
                    SearchResult(
                        content=result["content"],
                        importance=result["importance"],
                        created=result["created"],
                        memory_type="associative",
                        metadata=result.get("metadata", {}),
                        embedding=result.get("embedding"),
                        node_id=result.get("node_id"),
                        triplet=result.get("triplet"),
                    )
                    for result in a_mem_results
                ]
            )

            # Search spatial memory
            s_mem_results = self.persona.s_mem.search(
                terms=search_terms, triplet=query_triplet, time_range=time_range
            )
            results.extend(
                [
                    SearchResult(
                        content=result["content"],
                        importance=result["importance"],
                        created=result["created"],
                        memory_type="spatial",
                        metadata=result.get("metadata", {}),
                        embedding=result.get("embedding"),
                        node_id=result.get("node_id"),
                        triplet=result.get("triplet"),
                    )
                    for result in s_mem_results
                ]
            )

            # Search scratch memory
            scratch_results = self.persona.scratch.search(
                terms=search_terms, triplet=query_triplet, time_range=time_range
            )
            results.extend(
                [
                    SearchResult(
                        content=result["content"],
                        importance=result["importance"],
                        created=result["created"],
                        memory_type="scratch",
                        metadata=result.get("metadata", {}),
                        embedding=result.get("embedding"),
                        node_id=result.get("node_id"),
                        triplet=result.get("triplet"),
                    )
                    for result in scratch_results
                ]
            )

            return results
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def _update_last_accessed(self, results: List[SearchResult]) -> None:
        """Update last accessed time for retrieved memories."""
        try:
            current_time = datetime.now()
            for result in results:
                if result.node_id and result.memory_type == "associative":
                    self.persona.a_mem.update_last_accessed(
                        result.node_id, current_time
                    )
        except Exception as e:
            logger.error(f"Error updating last accessed time: {e}")

    def generate_focal_points(self, statements: str, n: int, persona: Any) -> List[str]:
        """Generate focal points using the LLM.

        Args:
            statements: Recent memories and thoughts to analyze
            n: Number of focal points to generate
            persona: The persona instance

        Returns:
            List of generated focal points (answers to theoretical questions)
        """
        try:
            prompt = f"""
            Given the following recent memories and thoughts:
            
            {statements}
            
            Generate {n} high-level insights that:
            1. Are grounded in the provided information
            2. Answer important theoretical questions about patterns or themes located in the provided information
            3. Provide meaningful conclusions or realizations
            4. Are specific and supported by the evidence
            
            Format each insight on a new line.
            """

            # Use the persona's LLM interface
            response = persona.llm.generate(
                prompt=prompt,
                temperature=0.7,  # Allow some creativity
                max_tokens=150,  # Enough for several insights
                stop=["\n\n"],  # Stop at double newline
            )

            # Clean and validate the response
            focal_points = [q.strip() for q in response.split("\n") if q.strip()]

            # Ensure we have the requested number of points
            if len(focal_points) < n:
                logger.warning(
                    f"Generated fewer focal points ({len(focal_points)}) than requested ({n})"
                )

            return focal_points[:n]

        except Exception as e:
            logger.error(f"Error generating focal points: {e}")
            return [
                "I've noticed patterns in my recent experiences that suggest..."
            ] * n  # Fallback focal point


class FocalPointGenerator:
    def __init__(self, persona: Any):
        self.persona = persona
        self.retrieve = Retrieve(persona)
        self.max_focal_points = 3
        self.min_importance = 0.3

    def generate_focal_points(self, n: Optional[int] = None) -> List[FocalPoint]:
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
