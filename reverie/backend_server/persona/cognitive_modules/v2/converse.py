"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: converse.py
Description: Defines the conversation generation and management module for generative agents.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import math

from persona.memory_structures.v2.associative_memory import AssociativeMemory
from persona.memory_structures.v2.scratch import Scratch
from persona.prompt_template.run_gpt_prompt import (
    run_gpt_prompt_agent_chat,
    run_gpt_prompt_agent_chat_summarize_ideas,
    run_gpt_prompt_agent_chat_summarize_relationship,
    run_gpt_prompt_chat_poignancy,
    run_gpt_prompt_event_poignancy,
    run_gpt_prompt_event_triple,
    run_gpt_prompt_generate_next_convo_line,
    run_gpt_prompt_generate_whisper_inner_thought,
    run_gpt_prompt_summarize_ideas,
    run_gpt_generate_iterative_chat_utt,
    run_gpt_generate_safety_score,
    run_gpt_prompt_summarize_conversation,
    run_gpt_prompt_planning_thought_on_convo,
    run_gpt_prompt_memo_on_convo,
)

logger = logging.getLogger(__name__)


@dataclass
class ChatState:
    """Represents the current state of a conversation."""

    messages: List[Tuple[str, str]]
    last_message_time: datetime
    is_active: bool = True
    chatting_with: Optional[str] = None
    chatting_end_time: Optional[datetime] = None
    chatting_with_buffer: Dict[str, int] = None

    def __post_init__(self):
        if self.chatting_with_buffer is None:
            self.chatting_with_buffer = {}

    def add_message(self, speaker: str, message: str):
        """Adds a new message to the conversation."""
        self.messages.append((speaker, message))
        self.last_message_time = datetime.now()

    def can_chat_with(self, target_name: str) -> bool:
        """Check if we can chat with the target persona."""
        if self.chatting_with:
            return False
        if target_name in self.chatting_with_buffer:
            return self.chatting_with_buffer[target_name] <= 0
        return True

    def update_buffer(self):
        """Update the chat buffer counts."""
        for name in list(self.chatting_with_buffer.keys()):
            if name != self.chatting_with:
                self.chatting_with_buffer[name] = max(
                    0, self.chatting_with_buffer[name] - 1
                )


class Converse:
    """Manages conversation generation and processing for generative agents."""

    def __init__(self):
        self.chat_states: Dict[str, ChatState] = {}

    def generate_agent_chat_summarize_ideas(
        self,
        init_persona: "Persona",
        target_persona: "Persona",
        retrieved: Dict[str, List["MemoryNode"]],
        curr_context: str,
    ) -> str:
        """Generates a summary of ideas for a conversation."""
        all_embedding_keys = []
        for nodes in retrieved.values():
            all_embedding_keys.extend(node.embedding_key for node in nodes)

        try:
            return run_gpt_prompt_agent_chat_summarize_ideas(
                init_persona,
                target_persona,
                "\n".join(all_embedding_keys),
                curr_context,
            )[0]
        except Exception as e:
            logger.error(f"Error generating chat summary: {e}")
            return ""

    def generate_summarize_agent_relationship(
        self,
        init_persona: "Persona",
        target_persona: "Persona",
        retrieved: Dict[str, List["MemoryNode"]],
    ) -> str:
        """Generates a summary of the relationship between two agents."""
        all_embedding_keys = []
        for nodes in retrieved.values():
            all_embedding_keys.extend(node.embedding_key for node in nodes)

        return run_gpt_prompt_agent_chat_summarize_relationship(
            init_persona, target_persona, "\n".join(all_embedding_keys)
        )[0]

    def generate_agent_chat(
        self,
        maze: "Maze",
        init_persona: "Persona",
        target_persona: "Persona",
        curr_context: str,
        init_summ_idea: str,
        target_summ_idea: str,
    ) -> List[Tuple[str, str]]:
        """Generates a conversation between two agents."""
        chat = run_gpt_prompt_agent_chat(
            maze,
            init_persona,
            target_persona,
            curr_context,
            init_summ_idea,
            target_summ_idea,
        )[0]

        for speaker, message in chat:
            logger.info(f"{speaker}: {message}")
        return chat

    def generate_one_utterance(
        self,
        maze: "Maze",
        init_persona: "Persona",
        target_persona: "Persona",
        retrieved: Dict[str, List["MemoryNode"]],
        curr_chat: List[Tuple[str, str]],
    ) -> Tuple[str, bool]:
        """Generates a single utterance in a conversation."""
        curr_context = (
            f"{init_persona.scratch.name} was {init_persona.scratch.act_description} "
            f"when {init_persona.scratch.name} saw {target_persona.scratch.name} "
            f"in the middle of {target_persona.scratch.act_description}.\n"
            f"{init_persona.scratch.name} is initiating a conversation with "
            f"{target_persona.scratch.name}."
        )

        try:
            result = run_gpt_generate_iterative_chat_utt(
                maze, init_persona, target_persona, retrieved, curr_context, curr_chat
            )[0]
            return result["utterance"], result["end"]
        except Exception as e:
            logger.error(f"Error generating utterance: {e}")
            return "", True

    def agent_chat(
        self,
        maze: "Maze",
        init_persona: "Persona",
        target_persona: "Persona",
        max_turns: int = 8,
    ) -> List[Tuple[str, str]]:
        """Manages a conversation between two agents."""
        # Validate conversation conditions
        if not self._validate_conversation_conditions(init_persona, target_persona):
            return []

        curr_chat = []
        logger.info("Starting agent chat")

        # Initialize chat states if needed
        if init_persona.scratch.name not in self.chat_states:
            self.chat_states[init_persona.scratch.name] = ChatState([], datetime.now())
        if target_persona.scratch.name not in self.chat_states:
            self.chat_states[target_persona.scratch.name] = ChatState(
                [], datetime.now()
            )

        # Check if we can chat with the target
        if not self.chat_states[init_persona.scratch.name].can_chat_with(
            target_persona.scratch.name
        ):
            logger.info(f"Cannot chat with {target_persona.scratch.name} due to buffer")
            return []

        # Set up chat states
        self.chat_states[init_persona.scratch.name].chatting_with = (
            target_persona.scratch.name
        )
        self.chat_states[target_persona.scratch.name].chatting_with = (
            init_persona.scratch.name
        )

        for _ in range(max_turns):
            # Generate utterance for init_persona
            focal_points = [f"{target_persona.scratch.name}"]
            retrieved = init_persona.retrieve(focal_points, 50)
            relationship = self.generate_summarize_agent_relationship(
                init_persona, target_persona, retrieved
            )

            last_chat = "\n".join(": ".join(msg) for msg in curr_chat[-4:])
            focal_points = [
                relationship,
                f"{target_persona.scratch.name} is {target_persona.scratch.act_description}",
            ]
            if last_chat:
                focal_points.append(last_chat)

            retrieved = init_persona.retrieve(focal_points, 15)
            utt, end = self.generate_one_utterance(
                maze, init_persona, target_persona, retrieved, curr_chat
            )

            curr_chat.append((init_persona.scratch.name, utt))
            if end:
                break

            # Generate utterance for target_persona
            focal_points = [f"{init_persona.scratch.name}"]
            retrieved = target_persona.retrieve(focal_points, 50)
            relationship = self.generate_summarize_agent_relationship(
                target_persona, init_persona, retrieved
            )

            last_chat = "\n".join(": ".join(msg) for msg in curr_chat[-4:])
            focal_points = [
                relationship,
                f"{init_persona.scratch.name} is {init_persona.scratch.act_description}",
            ]
            if last_chat:
                focal_points.append(last_chat)

            retrieved = target_persona.retrieve(focal_points, 15)
            utt, end = self.generate_one_utterance(
                maze, target_persona, init_persona, retrieved, curr_chat
            )

            curr_chat.append((target_persona.scratch.name, utt))
            if end:
                break

        # Update chat states
        self._update_chat_states(init_persona, target_persona, curr_chat)

        logger.info("Agent chat completed")
        return curr_chat

    def _validate_conversation_conditions(
        self, init_persona: "Persona", target_persona: "Persona"
    ) -> bool:
        """Validate if conversation conditions are met."""
        if not all(
            [
                init_persona.scratch.act_address,
                init_persona.scratch.act_description,
                target_persona.scratch.act_address,
                target_persona.scratch.act_description,
            ]
        ):
            return False

        if any(
            "sleeping" in desc
            for desc in [
                init_persona.scratch.act_description,
                target_persona.scratch.act_description,
            ]
        ):
            return False

        if init_persona.scratch.curr_time.hour == 23:
            return False

        if "<waiting>" in target_persona.scratch.act_address:
            return False

        return True

    def _update_chat_states(
        self,
        init_persona: "Persona",
        target_persona: "Persona",
        curr_chat: List[Tuple[str, str]],
    ) -> None:
        """Update chat states after conversation."""
        # Calculate conversation duration
        all_utt = "\n".join(f"{speaker}: {message}" for speaker, message in curr_chat)
        convo_length = math.ceil(int(len(all_utt) / 8) / 30)

        # Set end time
        end_time = init_persona.scratch.curr_time + timedelta(minutes=convo_length)

        # Update states
        self.chat_states[init_persona.scratch.name].chatting_end_time = end_time
        self.chat_states[target_persona.scratch.name].chatting_end_time = end_time

        # Update buffers
        self.chat_states[init_persona.scratch.name].chatting_with_buffer[
            target_persona.scratch.name
        ] = 800
        self.chat_states[target_persona.scratch.name].chatting_with_buffer[
            init_persona.scratch.name
        ] = 800

        # Generate and store conversation summaries
        self._store_conversation_summaries(init_persona, target_persona, curr_chat)

    def _store_conversation_summaries(
        self,
        init_persona: "Persona",
        target_persona: "Persona",
        curr_chat: List[Tuple[str, str]],
    ) -> None:
        """Generate and store conversation summaries."""
        all_utt = "\n".join(f"{speaker}: {message}" for speaker, message in curr_chat)

        # Generate summaries
        convo_summary = self.generate_convo_summary(init_persona, all_utt)
        planning_thought = self.generate_planning_thought_on_convo(
            init_persona, all_utt
        )
        memo = self.generate_memo_on_convo(init_persona, all_utt)

        # Store in memory
        self._store_in_memory(
            init_persona,
            convo_summary,
            planning_thought,
            memo,
            target_persona.scratch.name,
        )

        # Do the same for target persona
        convo_summary = self.generate_convo_summary(target_persona, all_utt)
        planning_thought = self.generate_planning_thought_on_convo(
            target_persona, all_utt
        )
        memo = self.generate_memo_on_convo(target_persona, all_utt)

        self._store_in_memory(
            target_persona,
            convo_summary,
            planning_thought,
            memo,
            init_persona.scratch.name,
        )

    def _store_in_memory(
        self,
        persona: "Persona",
        convo_summary: str,
        planning_thought: str,
        memo: str,
        other_persona: str,
    ) -> None:
        """Store conversation summaries in memory."""
        try:
            # Store conversation summary
            persona.a_mem.add_conversation(
                created=persona.scratch.curr_time,
                expiration=persona.scratch.curr_time + timedelta(days=30),
                description=convo_summary,
                other_persona=other_persona,
            )

            # Store planning thought
            persona.a_mem.add_thought(
                created=persona.scratch.curr_time,
                expiration=persona.scratch.curr_time + timedelta(days=30),
                description=planning_thought,
                thought_type="planning",
            )

            # Store memo
            persona.a_mem.add_thought(
                created=persona.scratch.curr_time,
                expiration=persona.scratch.curr_time + timedelta(days=30),
                description=memo,
                thought_type="memo",
            )
        except Exception as e:
            logger.error(f"Error storing conversation summaries: {e}")

    def generate_convo_summary(self, persona: "Persona", convo: str) -> str:
        """Generates a summary of a conversation."""
        return run_gpt_prompt_summarize_conversation(persona, convo)[0]

    def generate_planning_thought_on_convo(
        self, persona: "Persona", all_utt: str
    ) -> str:
        """Generates a planning thought based on a conversation."""
        return run_gpt_prompt_planning_thought_on_convo(persona, all_utt)[0]

    def generate_memo_on_convo(self, persona: "Persona", all_utt: str) -> str:
        """Generates a memo based on a conversation."""
        return run_gpt_prompt_memo_on_convo(persona, all_utt)[0]

    def generate_summarize_ideas(
        self, persona: "Persona", nodes: List["MemoryNode"], question: str
    ) -> str:
        """Generates a summary of ideas from memory nodes."""
        statements = "\n".join(node.embedding_key for node in nodes)
        return run_gpt_prompt_summarize_ideas(persona, statements, question)[0]

    def generate_next_line(
        self,
        persona: "Persona",
        interlocutor_desc: str,
        curr_convo: List[Tuple[str, str]],
        summarized_idea: str,
    ) -> str:
        """Generates the next line in a conversation."""
        prev_convo = "\n".join(
            f"{speaker}: {message}" for speaker, message in curr_convo
        )
        return run_gpt_prompt_generate_next_convo_line(
            persona, interlocutor_desc, prev_convo, summarized_idea
        )[0]

    def generate_inner_thought(self, persona: "Persona", whisper: str) -> str:
        """Generates an inner thought from a whisper."""
        return run_gpt_prompt_generate_whisper_inner_thought(persona, whisper)[0]

    def generate_action_event_triple(
        self, act_desp: str, persona: "Persona"
    ) -> Tuple[str, str, str]:
        """Generates a subject-predicate-object triple for an action."""
        return run_gpt_prompt_event_triple(act_desp, persona)[0]

    def generate_poig_score(
        self, persona: "Persona", event_type: str, description: str
    ) -> float:
        """Generates a poignancy score for an event."""
        if "is idle" in description:
            return 1.0

        if event_type in ["event", "thought"]:
            return run_gpt_prompt_event_poignancy(persona, description)[0]
        elif event_type == "chat":
            return run_gpt_prompt_chat_poignancy(
                persona, persona.scratch.act_description
            )[0]
        return 1.0

    def load_history_via_whisper(
        self, personas: Dict[str, "Persona"], whispers: List[Tuple[str, str]]
    ):
        """Loads history from whispers into persona memories."""
        for persona_name, whisper in whispers:
            persona = personas[persona_name]
            thought = self.generate_inner_thought(persona, whisper)

            created = persona.scratch.curr_time
            expiration = persona.scratch.curr_time + timedelta(days=30)
            s, p, o = self.generate_action_event_triple(thought, persona)
            keywords = {s, p, o}
            thought_poignancy = self.generate_poig_score(persona, "event", whisper)
            thought_embedding_pair = (thought, persona.get_embedding(thought))

            persona.a_mem.add_thought(
                created=created,
                expiration=expiration,
                subject=s,
                predicate=p,
                object=o,
                description=thought,
                keywords=keywords,
                poignancy=thought_poignancy,
                embedding_pair=thought_embedding_pair,
                filling=None,
            )

    def open_convo_session(self, persona: "Persona", convo_mode: str):
        """Opens a conversation session with a persona."""
        if convo_mode == "analysis":
            curr_convo = []
            interlocutor_desc = "Interviewer"

            while True:
                line = input("Enter Input: ")
                if line == "end_convo":
                    break

                if int(run_gpt_generate_safety_score(persona, line)[0]) >= 8:
                    print(
                        f"{persona.scratch.name} is a computational agent, and as such, "
                        "it may be inappropriate to attribute human agency to the agent "
                        "in your communication."
                    )
                else:
                    retrieved = persona.retrieve([line], 50)[line]
                    summarized_idea = self.generate_summarize_ideas(
                        persona, retrieved, line
                    )
                    curr_convo.append((interlocutor_desc, line))

                    next_line = self.generate_next_line(
                        persona, interlocutor_desc, curr_convo, summarized_idea
                    )
                    curr_convo.append((persona.scratch.name, next_line))

        elif convo_mode == "whisper":
            whisper = input("Enter Input: ")
            thought = self.generate_inner_thought(persona, whisper)

            created = persona.scratch.curr_time
            expiration = persona.scratch.curr_time + timedelta(days=30)
            s, p, o = self.generate_action_event_triple(thought, persona)
            keywords = {s, p, o}
            thought_poignancy = self.generate_poig_score(persona, "event", whisper)
            thought_embedding_pair = (thought, persona.get_embedding(thought))

            persona.a_mem.add_thought(
                created=created,
                expiration=expiration,
                subject=s,
                predicate=p,
                object=o,
                description=thought,
                keywords=keywords,
                poignancy=thought_poignancy,
                embedding_pair=thought_embedding_pair,
                filling=None,
            )
