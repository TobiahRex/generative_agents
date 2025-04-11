"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: associative_memory.py
Description: Defines the core long-term memory module for generative agents.
This is a cleaner, more maintainable version of the original implementation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    """Represents a single memory node in the associative memory system."""

    node_id: str
    node_count: int
    type_count: int
    node_type: str
    depth: int
    created: datetime
    expiration: Optional[datetime]
    subject: str
    predicate: str
    object: str
    description: str
    embedding_key: str
    poignancy: float
    keywords: Set[str]
    filling: Optional[List] = None
    last_accessed: datetime = None

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created

    def spo_summary(self) -> Tuple[str, str, str]:
        """Returns the subject-predicate-object summary of the node."""
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> Dict:
        """Converts the node to a dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_count": self.node_count,
            "type_count": self.type_count,
            "type": self.node_type,
            "depth": self.depth,
            "created": self.created.strftime("%Y-%m-%d %H:%M:%S"),
            "expiration": (
                self.expiration.strftime("%Y-%m-%d %H:%M:%S")
                if self.expiration
                else None
            ),
            "last_accessed": self.last_accessed.strftime("%Y-%m-%d %H:%M:%S"),
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "embedding_key": self.embedding_key,
            "poignancy": self.poignancy,
            "keywords": list(self.keywords),
            "filling": self.filling,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryNode":
        """Creates a MemoryNode from a dictionary."""
        return cls(
            node_id=data["node_id"],
            node_count=data["node_count"],
            type_count=data["type_count"],
            node_type=data["type"],
            depth=data["depth"],
            created=datetime.strptime(data["created"], "%Y-%m-%d %H:%M:%S"),
            expiration=(
                datetime.strptime(data["expiration"], "%Y-%m-%d %H:%M:%S")
                if data["expiration"]
                else None
            ),
            last_accessed=(
                datetime.strptime(data["last_accessed"], "%Y-%m-%d %H:%M:%S")
                if data.get("last_accessed")
                else None
            ),
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            description=data["description"],
            embedding_key=data["embedding_key"],
            poignancy=data["poignancy"],
            keywords=set(data["keywords"]),
            filling=data["filling"],
        )


class MemoryIndex:
    """Manages the indexing of memory nodes for efficient retrieval."""

    def __init__(self):
        self.kw_to_nodes: Dict[str, List[MemoryNode]] = {}
        self.kw_strength: Dict[str, int] = {}

    def add_node(self, node: MemoryNode, keywords: List[str]):
        """Adds a node to the index with its keywords."""
        for kw in keywords:
            kw = kw.lower()
            if kw in self.kw_to_nodes:
                self.kw_to_nodes[kw].insert(0, node)
            else:
                self.kw_to_nodes[kw] = [node]
            self.kw_strength[kw] = self.kw_strength.get(kw, 0) + 1

    def get_nodes_by_keyword(self, keyword: str) -> List[MemoryNode]:
        """Retrieves nodes associated with a keyword."""
        return self.kw_to_nodes.get(keyword.lower(), [])

    def get_keyword_strength(self, keyword: str) -> int:
        """Returns the strength of a keyword."""
        return self.kw_strength.get(keyword.lower(), 0)


class AssociativeMemory:
    """Manages the long-term memory system for generative agents."""

    def __init__(self, save_path: Union[str, Path]):
        self.save_path = Path(save_path)
        self.nodes: Dict[str, MemoryNode] = {}
        self.embeddings: Dict[str, List[float]] = {}

        # Initialize memory indices for different types
        self.event_index = MemoryIndex()
        self.thought_index = MemoryIndex()
        self.chat_index = MemoryIndex()

        # Sequential storage
        self.seq_event: List[MemoryNode] = []
        self.seq_thought: List[MemoryNode] = []
        self.seq_chat: List[MemoryNode] = []

        self._load_from_disk()

    def _load_from_disk(self):
        """Loads memory state from disk."""
        try:
            if not self.save_path.exists():
                return

            # Load embeddings
            embeddings_file = self.save_path / "embeddings.json"
            if embeddings_file.exists():
                self.embeddings = json.load(open(embeddings_file))

            # Load nodes
            nodes_file = self.save_path / "nodes.json"
            if nodes_file.exists():
                nodes_data = json.load(open(nodes_file))
                for node_id, node_data in nodes_data.items():
                    node = MemoryNode.from_dict(node_data)
                    self.nodes[node_id] = node

                    # Add to appropriate sequence and index
                    if node.node_type == "event":
                        self.seq_event.insert(0, node)
                        self.event_index.add_node(node, list(node.keywords))
                    elif node.node_type == "thought":
                        self.seq_thought.insert(0, node)
                        self.thought_index.add_node(node, list(node.keywords))
                    elif node.node_type == "chat":
                        self.seq_chat.insert(0, node)
                        self.chat_index.add_node(node, list(node.keywords))

            # Load keyword strengths
            kw_strength_file = self.save_path / "kw_strength.json"
            if kw_strength_file.exists():
                kw_strength_data = json.load(open(kw_strength_file))
                self.event_index.kw_strength = kw_strength_data.get(
                    "kw_strength_event", {}
                )
                self.thought_index.kw_strength = kw_strength_data.get(
                    "kw_strength_thought", {}
                )
        except Exception as e:
            logger.error(f"Error loading memory from disk: {e}")

    def save(self):
        """Saves memory state to disk."""
        try:
            self.save_path.mkdir(parents=True, exist_ok=True)

            # Save nodes
            nodes_data = {
                node_id: node.to_dict() for node_id, node in self.nodes.items()
            }
            with open(self.save_path / "nodes.json", "w") as f:
                json.dump(nodes_data, f)

            # Save embeddings
            with open(self.save_path / "embeddings.json", "w") as f:
                json.dump(self.embeddings, f)

            # Save keyword strengths
            kw_strength_data = {
                "kw_strength_event": self.event_index.kw_strength,
                "kw_strength_thought": self.thought_index.kw_strength,
            }
            with open(self.save_path / "kw_strength.json", "w") as f:
                json.dump(kw_strength_data, f)
        except Exception as e:
            logger.error(f"Error saving memory to disk: {e}")

    def add_memory(
        self,
        node_type: str,
        created: datetime,
        expiration: Optional[datetime],
        subject: str,
        predicate: str,
        object: str,
        description: str,
        keywords: Set[str],
        poignancy: float,
        embedding_pair: Tuple[str, List[float]],
        filling: Optional[List] = None,
    ) -> MemoryNode:
        """Adds a new memory node of the specified type."""
        try:
            node_count = len(self.nodes) + 1
            type_count = len(getattr(self, f"seq_{node_type}")) + 1
            node_id = f"node_{node_count}"

            # Calculate depth for thoughts
            depth = 1 if node_type == "thought" else 0
            if node_type == "thought" and filling:
                try:
                    depth += max(self.nodes[i].depth for i in filling)
                except (KeyError, ValueError):
                    pass

            # Clean up description for events
            if node_type == "event" and "(" in description:
                description = (
                    " ".join(description.split()[:3])
                    + " "
                    + description.split("(")[-1][:-1]
                )

            # Create node
            node = MemoryNode(
                node_id=node_id,
                node_count=node_count,
                type_count=type_count,
                node_type=node_type,
                depth=depth,
                created=created,
                expiration=expiration,
                subject=subject,
                predicate=predicate,
                object=object,
                description=description,
                embedding_key=embedding_pair[0],
                poignancy=poignancy,
                keywords=keywords,
                filling=filling,
            )

            # Store node
            self.nodes[node_id] = node
            getattr(self, f"seq_{node_type}").insert(0, node)
            getattr(self, f"{node_type}_index").add_node(node, list(keywords))
            self.embeddings[embedding_pair[0]] = embedding_pair[1]

            return node
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return None

    def add_event(self, *args, **kwargs) -> MemoryNode:
        """Adds a new event memory."""
        return self.add_memory("event", *args, **kwargs)

    def add_thought(self, *args, **kwargs) -> MemoryNode:
        """Adds a new thought memory."""
        return self.add_memory("thought", *args, **kwargs)

    def add_chat(self, *args, **kwargs) -> MemoryNode:
        """Adds a new chat memory."""
        return self.add_memory("chat", *args, **kwargs)

    def get_summarized_latest_events(self, retention: int) -> Set[Tuple[str, str, str]]:
        """Returns a set of the latest event summaries."""
        return {node.spo_summary() for node in self.seq_event[:retention]}

    def get_str_seq_events(self) -> str:
        """Returns a string representation of the event sequence."""
        return "\n".join(
            f"Event {len(self.seq_event) - i}: {node.spo_summary()} -- {node.description}"
            for i, node in enumerate(self.seq_event)
        )

    def get_str_seq_thoughts(self) -> str:
        """Returns a string representation of the thought sequence."""
        return "\n".join(
            f"Thought {len(self.seq_thought) - i}: {node.spo_summary()} -- {node.description}"
            for i, node in enumerate(self.seq_thought)
        )

    def get_str_seq_chats(self) -> str:
        """Returns a string representation of the chat sequence."""
        ret = []
        for node in self.seq_chat:
            ret.append(f"with {node.object} ({node.description})")
            ret.append(node.created.strftime("%B %d, %Y, %H:%M:%S"))
            ret.extend(f"{row[0]}: {row[1]}" for row in node.filling)
        return "\n".join(ret)

    def retrieve_relevant_thoughts(
        self, subject: str, predicate: str, object: str
    ) -> Set[MemoryNode]:
        """Retrieves thoughts relevant to the given subject, predicate, and object."""
        contents = [subject, predicate, object]
        relevant_nodes = set()
        for content in contents:
            relevant_nodes.update(
                self.thought_index.get_nodes_by_keyword(content.lower())
            )
        return relevant_nodes

    def retrieve_relevant_events(
        self, subject: str, predicate: str, object: str
    ) -> Set[MemoryNode]:
        """Retrieves events relevant to the given subject, predicate, and object."""
        contents = [subject, predicate, object]
        relevant_nodes = set()
        for content in contents:
            relevant_nodes.update(
                self.event_index.get_nodes_by_keyword(content.lower())
            )
        return relevant_nodes

    def get_last_chat(self, target_persona_name: str) -> Optional[MemoryNode]:
        """Returns the last chat with the specified persona."""
        chats = self.chat_index.get_nodes_by_keyword(target_persona_name.lower())
        return chats[0] if chats else None

    def get_keyword_strength(self, keyword: str, node_type: str = "event") -> int:
        """Returns the strength of a keyword for the specified node type."""
        index = getattr(self, f"{node_type}_index")
        return index.get_keyword_strength(keyword)

    def update_last_accessed(self, node_id: str):
        """Updates the last accessed time of a node."""
        if node_id in self.nodes:
            self.nodes[node_id].last_accessed = datetime.now()

    def prune_expired_memories(self):
        """Removes expired memories from the system."""
        current_time = datetime.now()
        for node_id, node in list(self.nodes.items()):
            if node.expiration and node.expiration < current_time:
                # Remove from sequences
                getattr(self, f"seq_{node.node_type}").remove(node)
                # Remove from indices
                for kw in node.keywords:
                    kw_nodes = getattr(self, f"{node.node_type}_index").kw_to_nodes.get(
                        kw.lower(), []
                    )
                    if node in kw_nodes:
                        kw_nodes.remove(node)
                # Remove from nodes
                del self.nodes[node_id]
