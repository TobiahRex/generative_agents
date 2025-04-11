"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: scratch.py
Description: Defines the short-term memory module for generative agents.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json


@dataclass
class Action:
    """Represents a current action being performed by the agent."""

    address: Optional[str]
    start_time: Optional[datetime]
    duration: int
    description: str
    pronunciatio: str
    event: Tuple[str, Optional[str], Optional[str]]
    path_set: bool = False
    planned_path: List[Tuple[int, int]] = None

    def __post_init__(self):
        if self.planned_path is None:
            self.planned_path = []


@dataclass
class ChatState:
    """Represents the current chat state of the agent."""

    chatting_with: Optional[str]
    chat: Optional[List[Tuple[str, str]]]
    chatting_with_buffer: Dict[str, int]
    chatting_end_time: Optional[datetime]


class Scratch:
    """Manages the short-term memory and current state of a generative agent."""

    def __init__(self, save_path: Union[str, Path]):
        # Persona configuration
        self.vision_r = 4
        self.att_bandwidth = 3
        self.retention = 5

        # World state
        self.curr_time: Optional[datetime] = None
        self.curr_tile: Optional[Tuple[int, int]] = None
        self.daily_plan_req: Optional[str] = None

        # Persona identity
        self.name: Optional[str] = None
        self.first_name: Optional[str] = None
        self.last_name: Optional[str] = None
        self.age: Optional[int] = None
        self.innate: Optional[str] = None
        self.learned: Optional[str] = None
        self.currently: Optional[str] = None
        self.lifestyle: Optional[str] = None
        self.living_area: Optional[str] = None

        # Reflection system
        self.concept_forget = 100
        self.daily_reflection_time = 60 * 3
        self.daily_reflection_size = 5
        self.overlap_reflect_th = 2
        self.kw_strg_event_reflect_th = 4
        self.kw_strg_thought_reflect_th = 4
        self.recency_w = 1
        self.relevance_w = 1
        self.importance_w = 1
        self.recency_decay = 0.99
        self.importance_trigger_max = 150
        self.importance_trigger_curr = self.importance_trigger_max
        self.importance_ele_n = 0
        self.thought_count = 5

        # Planning system
        self.daily_req: List[str] = []
        self.f_daily_schedule: List[Tuple[str, int]] = []
        self.f_daily_schedule_hourly_org: List[Tuple[str, int]] = []

        # Action management
        self.current_action = Action(None, None, 0, "", "", (None, None, None))
        self.chat_state = ChatState(None, None, {}, None)

        # Load saved state if exists
        self.save_path = Path(save_path)
        if self.save_path.exists():
            self._load_from_disk()

    def _load_from_disk(self):
        """Loads the scratch state from disk."""
        with open(self.save_path) as f:
            data = json.load(f)

        # Load basic configuration
        self.vision_r = data["vision_r"]
        self.att_bandwidth = data["att_bandwidth"]
        self.retention = data["retention"]

        # Load world state
        if data["curr_time"]:
            self.curr_time = datetime.strptime(data["curr_time"], "%B %d, %Y, %H:%M:%S")
        self.curr_tile = data["curr_tile"]
        self.daily_plan_req = data["daily_plan_req"]

        # Load persona identity
        self.name = data["name"]
        self.first_name = data["first_name"]
        self.last_name = data["last_name"]
        self.age = data["age"]
        self.innate = data["innate"]
        self.learned = data["learned"]
        self.currently = data["currently"]
        self.lifestyle = data["lifestyle"]
        self.living_area = data["living_area"]

        # Load reflection system
        self.concept_forget = data["concept_forget"]
        self.daily_reflection_time = data["daily_reflection_time"]
        self.daily_reflection_size = data["daily_reflection_size"]
        self.overlap_reflect_th = data["overlap_reflect_th"]
        self.kw_strg_event_reflect_th = data["kw_strg_event_reflect_th"]
        self.kw_strg_thought_reflect_th = data["kw_strg_thought_reflect_th"]
        self.recency_w = data["recency_w"]
        self.relevance_w = data["relevance_w"]
        self.importance_w = data["importance_w"]
        self.recency_decay = data["recency_decay"]
        self.importance_trigger_max = data["importance_trigger_max"]
        self.importance_trigger_curr = data["importance_trigger_curr"]
        self.importance_ele_n = data["importance_ele_n"]
        self.thought_count = data["thought_count"]

        # Load planning system
        self.daily_req = data["daily_req"]
        self.f_daily_schedule = data["f_daily_schedule"]
        self.f_daily_schedule_hourly_org = data["f_daily_schedule_hourly_org"]

        # Load action state
        self.current_action = Action(
            address=data["act_address"],
            start_time=(
                datetime.strptime(data["act_start_time"], "%B %d, %Y, %H:%M:%S")
                if data["act_start_time"]
                else None
            ),
            duration=data["act_duration"],
            description=data["act_description"],
            pronunciatio=data["act_pronunciatio"],
            event=tuple(data["act_event"]),
            path_set=data["act_path_set"],
            planned_path=data["planned_path"],
        )

        # Load chat state
        self.chat_state = ChatState(
            chatting_with=data["chatting_with"],
            chat=data["chat"],
            chatting_with_buffer=data["chatting_with_buffer"],
            chatting_end_time=(
                datetime.strptime(data["chatting_end_time"], "%B %d, %Y, %H:%M:%S")
                if data["chatting_end_time"]
                else None
            ),
        )

    def save(self):
        """Saves the current scratch state to disk."""
        data = {
            # Persona configuration
            "vision_r": self.vision_r,
            "att_bandwidth": self.att_bandwidth,
            "retention": self.retention,
            # World state
            "curr_time": (
                self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
                if self.curr_time
                else None
            ),
            "curr_tile": self.curr_tile,
            "daily_plan_req": self.daily_plan_req,
            # Persona identity
            "name": self.name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "innate": self.innate,
            "learned": self.learned,
            "currently": self.currently,
            "lifestyle": self.lifestyle,
            "living_area": self.living_area,
            # Reflection system
            "concept_forget": self.concept_forget,
            "daily_reflection_time": self.daily_reflection_time,
            "daily_reflection_size": self.daily_reflection_size,
            "overlap_reflect_th": self.overlap_reflect_th,
            "kw_strg_event_reflect_th": self.kw_strg_event_reflect_th,
            "kw_strg_thought_reflect_th": self.kw_strg_thought_reflect_th,
            "recency_w": self.recency_w,
            "relevance_w": self.relevance_w,
            "importance_w": self.importance_w,
            "recency_decay": self.recency_decay,
            "importance_trigger_max": self.importance_trigger_max,
            "importance_trigger_curr": self.importance_trigger_curr,
            "importance_ele_n": self.importance_ele_n,
            "thought_count": self.thought_count,
            # Planning system
            "daily_req": self.daily_req,
            "f_daily_schedule": self.f_daily_schedule,
            "f_daily_schedule_hourly_org": self.f_daily_schedule_hourly_org,
            # Action state
            "act_address": self.current_action.address,
            "act_start_time": (
                self.current_action.start_time.strftime("%B %d, %Y, %H:%M:%S")
                if self.current_action.start_time
                else None
            ),
            "act_duration": self.current_action.duration,
            "act_description": self.current_action.description,
            "act_pronunciatio": self.current_action.pronunciatio,
            "act_event": self.current_action.event,
            "act_path_set": self.current_action.path_set,
            "planned_path": self.current_action.planned_path,
            # Chat state
            "chatting_with": self.chat_state.chatting_with,
            "chat": self.chat_state.chat,
            "chatting_with_buffer": self.chat_state.chatting_with_buffer,
            "chatting_end_time": (
                self.chat_state.chatting_end_time.strftime("%B %d, %Y, %H:%M:%S")
                if self.chat_state.chatting_end_time
                else None
            ),
        }

        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_f_daily_schedule_index(self, advance: int = 0) -> int:
        """Returns the current index in the daily schedule based on elapsed time."""
        if not self.curr_time:
            return 0

        today_min_elapsed = self.curr_time.hour * 60 + self.curr_time.minute + advance
        elapsed = 0
        for i, (_, duration) in enumerate(self.f_daily_schedule):
            elapsed += duration
            if elapsed > today_min_elapsed:
                return i
        return len(self.f_daily_schedule)

    def get_f_daily_schedule_hourly_org_index(self, advance: int = 0) -> int:
        """Returns the current index in the hourly schedule based on elapsed time."""
        if not self.curr_time:
            return 0

        today_min_elapsed = self.curr_time.hour * 60 + self.curr_time.minute + advance
        elapsed = 0
        for i, (_, duration) in enumerate(self.f_daily_schedule_hourly_org):
            elapsed += duration
            if elapsed > today_min_elapsed:
                return i
        return len(self.f_daily_schedule_hourly_org)

    def get_str_iss(self) -> str:
        """Returns a string summary of the agent's identity stable set."""
        return (
            f"Name: {self.name}\n"
            f"Age: {self.age}\n"
            f"Innate traits: {self.innate}\n"
            f"Learned traits: {self.learned}\n"
            f"Currently: {self.currently}\n"
            f"Lifestyle: {self.lifestyle}\n"
            f"Daily plan requirement: {self.daily_plan_req}\n"
            f"Current Date: {self.curr_time.strftime('%A %B %d') if self.curr_time else 'Unknown'}\n"
        )

    def add_new_action(
        self,
        action_address: str,
        action_duration: int,
        action_description: str,
        action_pronunciatio: str,
        action_event: Tuple[str, str, str],
        chatting_with: Optional[str] = None,
        chat: Optional[List[Tuple[str, str]]] = None,
        chatting_with_buffer: Optional[Dict[str, int]] = None,
        chatting_end_time: Optional[datetime] = None,
        act_obj_description: Optional[str] = None,
        act_obj_pronunciatio: Optional[str] = None,
        act_obj_event: Optional[Tuple[str, str, str]] = None,
        act_start_time: Optional[datetime] = None,
    ):
        """Adds a new action to the current state."""
        self.current_action = Action(
            address=action_address,
            start_time=act_start_time or self.curr_time,
            duration=action_duration,
            description=action_description,
            pronunciatio=action_pronunciatio,
            event=action_event,
        )

        self.chat_state = ChatState(
            chatting_with=chatting_with,
            chat=chat,
            chatting_with_buffer=chatting_with_buffer or {},
            chatting_end_time=chatting_end_time,
        )

    def act_check_finished(self) -> bool:
        """Checks if the current action has finished."""
        if not self.current_action.address:
            return True

        if self.chat_state.chatting_with:
            end_time = self.chat_state.chatting_end_time
        else:
            if not self.current_action.start_time:
                return True
            end_time = self.current_action.start_time + timedelta(
                minutes=self.current_action.duration
            )

        return end_time and end_time <= self.curr_time

    def act_summarize(self) -> Dict:
        """Returns a dictionary summary of the current action."""
        return {
            "persona": self.name,
            "address": self.current_action.address,
            "start_datetime": self.current_action.start_time,
            "duration": self.current_action.duration,
            "description": self.current_action.description,
            "pronunciatio": self.current_action.pronunciatio,
        }

    def act_summary_str(self) -> str:
        """Returns a human-readable string summary of the current action."""
        if not self.current_action.start_time:
            return "No current action"

        start_datetime_str = self.current_action.start_time.strftime(
            "%A %B %d -- %H:%M %p"
        )
        return (
            f"[{start_datetime_str}]\n"
            f"Activity: {self.name} is {self.current_action.description}\n"
            f"Address: {self.current_action.address}\n"
            f"Duration in minutes: {self.current_action.duration} min\n"
        )

    def get_str_daily_schedule_summary(self) -> str:
        """Returns a formatted string of the daily schedule."""
        ret = []
        curr_min_sum = 0
        for task, duration in self.f_daily_schedule:
            curr_min_sum += duration
            hour = curr_min_sum // 60
            minute = curr_min_sum % 60
            ret.append(f"{hour:02}:{minute:02} || {task}")
        return "\n".join(ret)

    def get_str_daily_schedule_hourly_org_summary(self) -> str:
        """Returns a formatted string of the hourly schedule."""
        ret = []
        curr_min_sum = 0
        for task, duration in self.f_daily_schedule_hourly_org:
            curr_min_sum += duration
            hour = curr_min_sum // 60
            minute = curr_min_sum % 60
            ret.append(f"{hour:02}:{minute:02} || {task}")
        return "\n".join(ret)
