from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path

from src.config import (
    ACTION_NAMES,
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    LEARNING_RATE,
)


@dataclass
class AgentConfig:
    learning_rate: float = LEARNING_RATE
    discount_factor: float = DISCOUNT_FACTOR
    epsilon: float = EPSILON_START
    epsilon_min: float = EPSILON_MIN
    epsilon_decay: float = EPSILON_DECAY
    learning_enabled: bool = True
    heuristics_enabled: bool = False


class QLearningAgent:
    """Q-learning agent for Snake reinforcement learning.

    Implements tabular Q-learning with epsilon-greedy exploration and optional
    heuristic-guided action selection.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the Q-learning agent.

        Args:
            config: Agent configuration. Uses default if None.
        """
        self.config = config or AgentConfig()
        self.q_table: dict[str, list[float]] = {}

    def choose_action(self, state: tuple[str, ...], explore: bool = True) -> int:
        """Select an action using epsilon-greedy or heuristic strategy.

        Args:
            state: Current game state representation.
            explore: If True, use epsilon-greedy exploration.

        Returns:
            Action index (0: UP, 1: LEFT, 2: DOWN, 3: RIGHT).
        """
        state_key = self._encode_state(state)
        if not self.config.heuristics_enabled:
            if explore and self.config.epsilon > 0 and random.random() < self.config.epsilon:
                return random.randrange(len(ACTION_NAMES))

            q_values = self.q_table.get(state_key)
            if q_values is None:
                q_values = [0.0] * len(ACTION_NAMES)

            best_value = max(q_values)
            best_actions = [index for index, value in enumerate(q_values) if value == best_value]
            return random.choice(best_actions)

        state_info = self._parse_state(state)
        forbidden_action = self._opposite_action(state_info["last_action"])

        if explore and self.config.epsilon > 0 and random.random() < self.config.epsilon:
            return self._heuristic_explore(state_info, forbidden_action)

        q_values = self.q_table.get(state_key)
        if q_values is None:
            q_values = [0.0] * len(ACTION_NAMES)

        candidate_actions = [index for index in range(len(ACTION_NAMES)) if index != forbidden_action]
        best_value = max(q_values[index] for index in candidate_actions)
        best_actions = [index for index in candidate_actions if q_values[index] == best_value]
        if len(best_actions) > 1:
            return self._heuristic_choice(state_info, forbidden_action, best_actions)
        return random.choice(best_actions)

    def update(
        self,
        state: tuple[str, ...],
        action: int,
        reward: float,
        next_state: tuple[str, ...],
        done: bool,
    ) -> None:
        """Update Q-values using the Q-learning update rule.

        Args:
            state: Current game state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether the episode terminated.
        """
        if not self.config.learning_enabled:
            return

        state_key = self._encode_state(state)
        next_key = self._encode_state(next_state)
        current_q = self.q_table.setdefault(state_key, [0.0] * len(ACTION_NAMES))
        next_q = self.q_table.setdefault(next_key, [0.0] * len(ACTION_NAMES))

        target = reward
        if not done:
            target += self.config.discount_factor * max(next_q)

        current_q[action] += self.config.learning_rate * (target - current_q[action])

    def decay_exploration(self) -> None:
        """Reduce epsilon to decrease exploration over time."""
        self.config.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon * self.config.epsilon_decay,
        )

    def save(self, file_path: str | Path) -> None:
        """Save the agent's configuration and Q-table to JSON.

        Args:
            file_path: Path to save the model.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "config": asdict(self.config),
            "q_table": self.q_table,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self, file_path: str | Path) -> None:
        """Load the agent's configuration and Q-table from JSON.

        Args:
            file_path: Path to the saved model.
        """
        path = Path(file_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        config_data = payload.get("config", {})
        self.config = AgentConfig(**{**asdict(self.config), **config_data})
        self.q_table = {key: list(values) for key, values in payload.get("q_table", {}).items()}

    @staticmethod
    def _encode_state(state: tuple[str, ...]) -> str:
        """Convert state tuple to a string key for Q-table lookup."""
        return "|".join(state)

    @staticmethod
    def _extract_last_action(state: tuple[str, ...]) -> int:
        """Extract the last action from the state representation."""
        last_part = state[-1]
        if last_part.startswith("A:"):
            try:
                return int(last_part[2:])
            except ValueError:
                return 3
        return 3

    @staticmethod
    def _opposite_action(action: int) -> int:
        return {0: 2, 1: 3, 2: 0, 3: 1}.get(action, 3)

    def _parse_state(self, state: tuple[str, ...]) -> dict[str, object]:
        rays: list[tuple[str, int]] = []
        for part in state[:4]:
            if ":" in part:
                symbol, dist = part.split(":", 1)
                try:
                    distance = int(dist)
                except ValueError:
                    distance = 999
                rays.append((symbol, distance))
            else:
                rays.append(("W", 999))

        food_dir = None
        food_dx = 0
        food_dy = 0
        for part in state:
            if part.startswith("F:"):
                try:
                    food_dir = int(part[2:])
                except ValueError:
                    food_dir = None
            elif part.startswith("GX:"):
                try:
                    food_dx = int(part[3:])
                except ValueError:
                    food_dx = 0
            elif part.startswith("GY:"):
                try:
                    food_dy = int(part[3:])
                except ValueError:
                    food_dy = 0

        return {
            "rays": rays,
            "food_dir": food_dir,
            "food_dx": food_dx,
            "food_dy": food_dy,
            "last_action": self._extract_last_action(state),
        }

    def _heuristic_explore(self, state_info: dict[str, object], forbidden_action: int) -> int:
        choices = [action for action in range(len(ACTION_NAMES)) if action != forbidden_action]
        return self._heuristic_choice(state_info, forbidden_action, choices)

    def _heuristic_choice(self, state_info: dict[str, object], forbidden_action: int, choices: list[int]) -> int:
        rays = state_info["rays"]
        food_dir = state_info["food_dir"]
        weights: list[float] = []

        for action in choices:
            if action == forbidden_action:
                weights.append(0.0)
                continue

            symbol, distance = rays[action]
            weight = 1.0
            if distance == 1 and symbol in {"W", "S"}:
                weight = 0.01
            elif distance == 1 and symbol == "R":
                weight = 0.1

            if food_dir is not None and food_dir >= 0 and action == food_dir:
                weight *= 2.5

            weights.append(weight)

        if sum(weights) == 0:
            return random.choice(choices)

        return random.choices(choices, weights=weights, k=1)[0]
