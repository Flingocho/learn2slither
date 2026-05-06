from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random
from typing import Iterable

from snake_rl.config import (
    ACTIONS,
    ACTION_NAMES,
    BOARD_SIZE,
    DEATH_REWARD,
    GREEN_APPLES,
    GREEN_REWARD,
    INITIAL_SNAKE_LENGTH,
    MAX_STEPS_WITHOUT_PROGRESS,
    RED_APPLES,
    RED_REWARD,
    STEP_REWARD,
)


@dataclass(frozen=True)
class StepResult:
    state: tuple[str, ...]
    reward: float
    done: bool
    info: dict[str, object]


class SnakeEnvironment:
    def __init__(self, heuristics_enabled: bool = False, board_size: int | None = None) -> None:
        self.board_size = board_size if board_size is not None else BOARD_SIZE
        self.snake: list[tuple[int, int]] = []
        self.green_apples: set[tuple[int, int]] = set()
        self.red_apples: set[tuple[int, int]] = set()
        self.score = 0
        self.steps = 0
        self.steps_since_progress = 0
        self.done = False
        self.game_over_reason = ""
        self.last_action = 3  # START facing RIGHT
        self.heuristics_enabled = heuristics_enabled
        self.recent_positions: deque[tuple[int, int]] = deque(maxlen=12)
        self.reset()

    def reset(self, seed: int | None = None) -> tuple[str, ...]:
        if seed is not None:
            random.seed(seed)

        self.snake = self._spawn_snake()
        self.green_apples = set()
        self.red_apples = set()
        self.score = 0
        self.steps = 0
        self.steps_since_progress = 0
        self.done = False
        self.game_over_reason = ""
        self.last_action = 3  # Reset to facing RIGHT
        self.recent_positions.clear()
        self._spawn_apples()
        return self.get_state()

    @property
    def head(self) -> tuple[int, int]:
        return self.snake[0]

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(self.get_state(), 0.0, True, {"reason": self.game_over_reason})

        self.last_action = action
        self.steps += 1
        self.steps_since_progress += 1

        distance_before = self._distance_to_nearest_green(self.head)
        dx, dy = ACTIONS[action]
        next_head = (self.head[0] + dx, self.head[1] + dy)
        reward = STEP_REWARD
        ate_green = False
        ate_red = False

        if not self._in_bounds(next_head):
            return self._end_episode(next_head, reward + DEATH_REWARD, "hit_wall")

        will_grow = next_head in self.green_apples
        will_shrink = next_head in self.red_apples

        body_collision = next_head in self.snake
        if body_collision and not (not will_grow and next_head == self.snake[-1]):
            return self._end_episode(next_head, reward + DEATH_REWARD, "hit_self")

        self.snake.insert(0, next_head)

        if self.heuristics_enabled and next_head in self.recent_positions:
            reward -= 1.0
        if self.heuristics_enabled:
            self.recent_positions.append(next_head)

        if will_grow:
            ate_green = True
            reward += GREEN_REWARD
            self.score += 1
            self.steps_since_progress = 0
            self.green_apples.remove(next_head)
        elif will_shrink:
            ate_red = True
            reward += RED_REWARD
            self.score -= 1
            self.steps_since_progress = 0
            self.red_apples.remove(next_head)
            self.snake.pop()
            if len(self.snake) == 0:
                return self._end_episode(next_head, reward + DEATH_REWARD, "length_zero")
        else:
            self.snake.pop()

        distance_after = self._distance_to_nearest_green(self.head)
        if distance_before is not None and distance_after is not None and not ate_green and not ate_red:
            reward += (distance_before - distance_after) * 3.0

        self._spawn_apples()

        if self.steps_since_progress >= MAX_STEPS_WITHOUT_PROGRESS:
            return self._end_episode(next_head, reward + DEATH_REWARD, "timeout")

        state = self.get_state()
        info = {
            "reason": "",
            "action_name": ACTION_NAMES[action],
            "vision": self.describe_vision(),
            "ate_green": ate_green,
            "ate_red": ate_red,
        }
        return StepResult(state, reward, False, info)

    def get_state(self) -> tuple[str, ...]:
        state_parts = []
        for direction in range(4):
            distance = 0
            symbol = "W"
            for cell in self._ray_positions(direction):
                distance += 1
                cell_sym = self._cell_symbol(cell)
                if cell_sym != "0":
                    symbol = cell_sym
                    break
            discretized_distance = self._discretize_distance(distance)
            state_parts.append(f"{symbol}:{discretized_distance}")
        
        if self.heuristics_enabled:
            # Dirección a la manzana verde más cercana
            direction_to_food = self._direction_to_nearest_green()
            state_parts.append(f"F:{direction_to_food}")

            distance_to_food = self._distance_to_nearest_green(self.head)
            state_parts.append(f"D:{distance_to_food if distance_to_food is not None else -1}")

            food_dx, food_dy = self._relative_food_vector()
            state_parts.append(f"GX:{food_dx}")
            state_parts.append(f"GY:{food_dy}")

            # Agregar la última acción realizada como parte del estado
            state_parts.append(f"A:{self.last_action}")
        return tuple(state_parts)

    def _direction_to_nearest_green(self) -> int:
        """Retorna la dirección (0-3) hacia la manzana verde más cercana, o -1 si no hay."""
        if not self.green_apples:
            return -1
        
        hx, hy = self.head
        nearest = None
        min_dist = float('inf')
        
        for gx, gy in self.green_apples:
            dist = abs(gx - hx) + abs(gy - hy)
            if dist < min_dist:
                min_dist = dist
                nearest = (gx, gy)
        
        if nearest is None:
            return -1
        
        gx, gy = nearest
        dx = gx - hx
        dy = gy - hy
        
        # Determinar la dirección general
        # 0=UP, 1=LEFT, 2=DOWN, 3=RIGHT
        if abs(dy) > abs(dx):
            return 0 if dy < 0 else 2  # UP o DOWN
        else:
            return 1 if dx < 0 else 3  # LEFT o RIGHT

    def _distance_to_nearest_green(self, origin: tuple[int, int]) -> int | None:
        if not self.green_apples:
            return None

        ox, oy = origin
        nearest_distance = None
        for gx, gy in self.green_apples:
            distance = abs(gx - ox) + abs(gy - oy)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
        return nearest_distance

    def _relative_food_vector(self) -> tuple[int, int]:
        if not self.green_apples:
            return 0, 0

        hx, hy = self.head
        nearest_food = min(
            self.green_apples,
            key=lambda cell: abs(cell[0] - hx) + abs(cell[1] - hy),
        )
        fx, fy = nearest_food
        return fx - hx, fy - hy

    def describe_vision(self) -> list[str]:
        return [self._ray_line(direction) for direction in range(4)]

    def render_snapshot(self) -> list[list[str]]:
        board = [["0" for _ in range(self.board_size)] for _ in range(self.board_size)]
        for x, y in self.green_apples:
            board[y][x] = "G"
        for x, y in self.red_apples:
            board[y][x] = "R"
        for index, (x, y) in enumerate(self.snake):
            board[y][x] = "H" if index == 0 else "S"
        return board

    def _spawn_snake(self) -> list[tuple[int, int]]:
        directions = [0, 1, 2, 3]
        while True:
            direction = random.choice(directions)
            dx, dy = ACTIONS[direction]
            dx = -dx
            dy = -dy

            x = random.randrange(self.board_size)
            y = random.randrange(self.board_size)
            body = [(x + dx * index, y + dy * index) for index in range(INITIAL_SNAKE_LENGTH)]
            if all(self._in_bounds(cell) for cell in body):
                return body

    def _spawn_apples(self) -> None:
        while len(self.green_apples) < GREEN_APPLES:
            self.green_apples.add(self._random_empty_cell())

        while len(self.red_apples) < RED_APPLES:
            self.red_apples.add(self._random_empty_cell())

    def _random_empty_cell(self) -> tuple[int, int]:
        occupied = set(self.snake) | self.green_apples | self.red_apples
        available = [
            (x, y)
            for y in range(self.board_size)
            for x in range(self.board_size)
            if (x, y) not in occupied
        ]
        if not available:
            raise RuntimeError("No free cells left to spawn apples.")
        return random.choice(available)

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def _cell_symbol(self, cell: tuple[int, int]) -> str:
        if cell == self.head:
            return "H"
        if cell in self.snake:
            return "S"
        if cell in self.green_apples:
            return "G"
        if cell in self.red_apples:
            return "R"
        return "0"

    @staticmethod
    def _discretize_distance(distance: int) -> str:
        if distance == 1:
            return "1"
        elif distance <= 2:
            return "2"
        elif distance <= 4:
            return "3"
        elif distance <= 6:
            return "4"
        else:
            return "5"

    def _ray_positions(self, direction: int) -> Iterable[tuple[int, int]]:
        dx, dy = ACTIONS[direction]
        x, y = self.head
        while True:
            x += dx
            y += dy
            if not self._in_bounds((x, y)):
                break
            yield x, y

    def _ray_line(self, direction: int) -> str:
        line = ["H"]
        line.extend(self._cell_symbol(cell) for cell in self._ray_positions(direction))
        line.append("W")
        return "".join(line)

    def _ray_signature(self, direction: int) -> str:
        distance = 0
        for cell in self._ray_positions(direction):
            distance += 1
            symbol = self._cell_symbol(cell)
            if symbol != "0":
                return f"{symbol}:{distance}"
        return f"W:{distance + 1}"

    def _end_episode(self, location: tuple[int, int], reward: float, reason: str) -> StepResult:
        self.done = True
        self.game_over_reason = reason
        state = self.get_state()
        info = {
            "reason": reason,
            "collision": location,
            "vision": self.describe_vision(),
            "action_name": "",
            "ate_green": False,
            "ate_red": False,
        }
        return StepResult(state, reward, True, info)
