# Cognitive Modules v2

This directory contains the v2 implementation of the cognitive modules for generative agents. The implementation is cleaner, more maintainable, and follows better OOP principles.

## Modules

### Converse (converse.py)

The `Converse` module handles conversation generation and management for generative agents. It provides:

1. **Conversation Generation**
   - Agent-to-agent conversations
   - Conversation summarization
   - Relationship analysis
   - Context-aware responses

2. **Conversation Management**
   - Chat session handling
   - Conversation state tracking
   - Memory integration
   - Safety checks

3. **Thought Generation**
   - Inner thought processing
   - Whisper interpretation
   - Action event generation
   - Poignancy scoring

4. **Analysis Tools**
   - Relationship summarization
   - Idea summarization
   - Context analysis
   - Safety scoring

### Perceive (perceive.py)

The `Perceive` module handles perception and awareness for generative agents. It provides:

1. **Event Perception**
   - Event detection and processing
   - Event importance scoring
   - Event memory storage
   - Event context analysis

2. **Spatial Awareness**
   - Location tracking
   - Object detection
   - Spatial relationships
   - Movement awareness

3. **Memory Integration**
   - Event-to-memory conversion
   - Memory importance weighting
   - Memory decay management
   - Memory retrieval optimization

4. **State Management**
   - Current state tracking
   - State transitions
   - State persistence
   - State validation

### Plan (plan.py)

The `Plan` module handles planning and scheduling for generative agents. It provides:

1. **Daily Planning**
   - Daily plan generation
   - Schedule optimization
   - Activity prioritization
   - Time management

2. **Action Planning**
   - Action sequence generation
   - Action validation
   - Action execution tracking
   - Action completion monitoring

3. **Goal Management**
   - Goal setting and tracking
   - Goal prioritization
   - Goal progress monitoring
   - Goal achievement validation

4. **Schedule Management**
   - Schedule creation and updates
   - Time slot allocation
   - Conflict resolution
   - Schedule optimization

## Key Improvements in v2

1. **Better Code Organization**
   - Clear separation of concerns
   - Modular design
   - Type safety with type hints
   - Class-based structure

2. **Improved Documentation**
   - Clear docstrings
   - External documentation (this README)
   - Better code readability
   - Usage examples

3. **Enhanced Maintainability**
   - Reduced code duplication
   - Better error handling
   - More consistent interfaces
   - Cleaner function signatures

4. **Modern Python Features**
   - Type hints
   - Dataclasses for data structures
   - Better error handling
   - Improved logging

## Usage

The modules provide comprehensive tools for managing agent behavior:

```python
from converse import Converse
from perceive import Perceive
from plan import Plan

# Initialize modules
converse = Converse()
perceive = Perceive()
plan = Plan()

# Handle perception
events = perceive.process_events(
    persona=persona,
    maze=maze,
    events=events
)

# Generate agent chat
chat = converse.agent_chat(
    maze=maze,
    init_persona=persona1,
    target_persona=persona2
)

# Generate daily plan
daily_plan = plan.generate_daily_plan(
    persona=persona,
    maze=maze
)

# Generate action sequence
action_sequence = plan.generate_action_sequence(
    persona=persona,
    maze=maze
)
```

## Design Patterns

The v2 implementation uses several design patterns to improve code organization and maintainability:

1. **Strategy Pattern**
   - Different conversation generation strategies
   - Configurable safety checks
   - Extensible summarization methods
   - Event processing strategies
   - Planning strategies

2. **State Pattern**
   - Conversation state management
   - Memory integration
   - Context tracking
   - Perception state management
   - Planning state management

3. **Factory Pattern**
   - Conversation object creation
   - Memory node generation
   - Event triple generation
   - Event object creation
   - Plan object creation

4. **Observer Pattern**
   - Memory updates
   - State changes
   - Event notifications
   - Perception updates
   - Plan updates 