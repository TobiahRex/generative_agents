"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: spatial_memory.py
Description: Defines the spatial memory module for generative agents.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import json


@dataclass
class Location:
    """Represents a location in the game world."""

    world: str
    sector: Optional[str] = None
    arena: Optional[str] = None
    game_objects: List[str] = None

    def __post_init__(self):
        if self.game_objects is None:
            self.game_objects = []

    def __str__(self) -> str:
        parts = [self.world]
        if self.sector:
            parts.append(self.sector)
        if self.arena:
            parts.append(self.arena)
        return ":".join(parts)


class SpatialMemory:
    """Manages the spatial memory and navigation information for generative agents."""

    def __init__(self, save_path: Union[str, Path]):
        self.tree: Dict = {}
        self.save_path = Path(save_path)
        if self.save_path.exists():
            self._load_from_disk()

    def _load_from_disk(self):
        """Loads the spatial memory tree from disk."""
        with open(self.save_path) as f:
            self.tree = json.load(f)

    def save(self):
        """Saves the spatial memory tree to disk."""
        with open(self.save_path, "w") as f:
            json.dump(self.tree, f, indent=2)

    def print_tree(self):
        """Prints the spatial memory tree in a hierarchical format."""

        def _print_tree(tree: Union[Dict, List], depth: int = 0):
            dash = " >" * depth
            if isinstance(tree, list):
                if tree:
                    print(f"{dash} {tree}")
                return

            for key, val in tree.items():
                if key:
                    print(f"{dash} {key}")
                _print_tree(val, depth + 1)

        _print_tree(self.tree)

    def get_str_accessible_sectors(self, curr_world: str) -> str:
        """
        Returns a comma-separated string of all accessible sectors in the current world.

        Args:
            curr_world: The current world name

        Returns:
            A string of accessible sectors, e.g., "bedroom, kitchen, dining room"
        """
        if curr_world not in self.tree:
            return ""
        return ", ".join(list(self.tree[curr_world].keys()))

    def get_str_accessible_sector_arenas(self, sector: str) -> str:
        """
        Returns a comma-separated string of all accessible arenas in the current sector.

        Args:
            sector: The sector address in format "world:sector"

        Returns:
            A string of accessible arenas, e.g., "bedroom, kitchen, dining room"
        """
        try:
            world, sector_name = sector.split(":")
            if (
                not sector_name
                or world not in self.tree
                or sector_name not in self.tree[world]
            ):
                return ""
            return ", ".join(list(self.tree[world][sector_name].keys()))
        except ValueError:
            return ""

    def get_str_accessible_arena_game_objects(self, arena: str) -> str:
        """
        Returns a comma-separated string of all accessible game objects in the current arena.

        Args:
            arena: The arena address in format "world:sector:arena"

        Returns:
            A string of accessible game objects, e.g., "phone, charger, bed"
        """
        try:
            world, sector, arena_name = arena.split(":")
            if not arena_name:
                return ""

            # Try both original and lowercase arena names
            try:
                objects = self.tree[world][sector][arena_name]
            except KeyError:
                objects = self.tree[world][sector][arena_name.lower()]

            return ", ".join(objects)
        except (ValueError, KeyError):
            return ""

    def get_location(self, address: str) -> Optional[Location]:
        """
        Returns a Location object for the given address.

        Args:
            address: The location address in format "world:sector:arena"

        Returns:
            A Location object or None if the address is invalid
        """
        try:
            parts = address.split(":")
            if len(parts) < 1:
                return None

            world = parts[0]
            sector = parts[1] if len(parts) > 1 else None
            arena = parts[2] if len(parts) > 2 else None

            if world not in self.tree:
                return None

            if sector and sector not in self.tree[world]:
                return None

            if arena and sector and arena not in self.tree[world][sector]:
                return None

            game_objects = []
            if arena and sector:
                try:
                    game_objects = self.tree[world][sector][arena]
                except KeyError:
                    try:
                        game_objects = self.tree[world][sector][arena.lower()]
                    except KeyError:
                        pass

            return Location(world, sector, arena, game_objects)
        except (ValueError, KeyError):
            return None

    def is_accessible(self, address: str) -> bool:
        """
        Checks if a location is accessible.

        Args:
            address: The location address in format "world:sector:arena"

        Returns:
            True if the location is accessible, False otherwise
        """
        location = self.get_location(address)
        return location is not None

    def get_accessible_locations(self, current_address: str) -> List[str]:
        """
        Returns a list of accessible locations from the current address.

        Args:
            current_address: The current location address

        Returns:
            A list of accessible location addresses
        """
        current = self.get_location(current_address)
        if not current:
            return []

        accessible = []

        # Add all sectors in the current world
        if current.sector is None:
            for sector in self.tree[current.world].keys():
                accessible.append(f"{current.world}:{sector}")
            return accessible

        # Add all arenas in the current sector
        if current.arena is None:
            for arena in self.tree[current.world][current.sector].keys():
                accessible.append(f"{current.world}:{current.sector}:{arena}")
            return accessible

        # Current location is an arena, return its game objects
        return [
            f"{current.world}:{current.sector}:{current.arena}:{obj}"
            for obj in current.game_objects
        ]
