from __future__ import annotations

from dataclasses import dataclass

import pygame

from snake_rl.config import (
    BACKGROUND,
    BOARD_BG,
    CELL_SIZE,
    GREEN_APPLE,
    GRID_LINE,
    HIGHLIGHT,
    MARGIN,
    PANEL_BG,
    PANEL_HEIGHT,
    RED_APPLE,
    SNAKE_BODY,
    SNAKE_HEAD,
    SNAKE_OUTLINE,
    SUBTEXT,
    TEXT,
    WARNING,
)


@dataclass
class RenderSnapshot:
    episode: int
    step: int
    score: int
    epsilon: float
    speed_label: str
    action_name: str
    reward: float
    state: tuple[str, ...]
    vision: list[str]
    reason: str
    paused: bool


class SnakeRenderer:
    def __init__(self, board_size: int = 10) -> None:
        pygame.init()
        self.board_size = board_size
        self.window_width = board_size * CELL_SIZE + MARGIN * 2
        self.window_height = board_size * CELL_SIZE + PANEL_HEIGHT + MARGIN * 2
        pygame.display.set_caption("Learn2Slither")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("dejavusans", 28, bold=True)
        self.font_body = pygame.font.SysFont("dejavusans", 20)
        self.font_small = pygame.font.SysFont("dejavusans", 16)

    def close(self) -> None:
        pygame.quit()

    def pump_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def wait_for_step(self) -> bool:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_n):
                        return True
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        return True
            self.clock.tick(30)

    def render(self, env, snapshot: RenderSnapshot) -> None:
        self.screen.fill(BACKGROUND)
        self._draw_panel(snapshot)
        self._draw_board(env)
        pygame.display.flip()
        self.clock.tick(60)

    def _draw_panel(self, snapshot: RenderSnapshot) -> None:
        panel_rect = pygame.Rect(0, 0, self.window_width, PANEL_HEIGHT + MARGIN)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)

        title = self.font_title.render("Learn2Slither", True, TEXT)
        self.screen.blit(title, (MARGIN, 12))

        info_left = [
            f"Episode: {snapshot.episode}",
            f"Step: {snapshot.step}",
            f"Score: {snapshot.score}",
        ]
        info_right = [
            f"Epsilon: {snapshot.epsilon:.3f}",
            f"Speed: {snapshot.speed_label}",
            f"Action: {snapshot.action_name or '-'}",
        ]

        for index, text in enumerate(info_left):
            surface = self.font_body.render(text, True, TEXT)
            self.screen.blit(surface, (MARGIN, 52 + index * 24))

        for index, text in enumerate(info_right):
            surface = self.font_body.render(text, True, TEXT)
            self.screen.blit(surface, (self.window_width // 2 + 24, 52 + index * 24))

        reward = self.font_body.render(f"Reward: {snapshot.reward:+.2f}", True, HIGHLIGHT if snapshot.reward >= 0 else WARNING)
        self.screen.blit(reward, (MARGIN, 120))

        status_text = snapshot.reason if snapshot.reason else ("Paused" if snapshot.paused else "Running")
        status_color = WARNING if snapshot.reason else (SUBTEXT if snapshot.paused else TEXT)
        status = self.font_body.render(f"Status: {status_text}", True, status_color)
        self.screen.blit(status, (self.window_width // 2 + 24, 120))

    def _draw_board(self, env) -> None:
        board_origin_y = PANEL_HEIGHT + MARGIN
        board_rect = pygame.Rect(MARGIN, board_origin_y, self.board_size * CELL_SIZE, self.board_size * CELL_SIZE)
        pygame.draw.rect(self.screen, BOARD_BG, board_rect, border_radius=12)

        for row in range(self.board_size):
            for col in range(self.board_size):
                rect = pygame.Rect(
                    MARGIN + col * CELL_SIZE + 1,
                    board_origin_y + row * CELL_SIZE + 1,
                    CELL_SIZE - 2,
                    CELL_SIZE - 2,
                )
                pygame.draw.rect(self.screen, GRID_LINE, rect, width=1, border_radius=6)

        for x, y in env.green_apples:
            self._draw_apple(x, y, board_origin_y, GREEN_APPLE)
        for x, y in env.red_apples:
            self._draw_apple(x, y, board_origin_y, RED_APPLE)

        if env.snake:
            for i, (x, y) in enumerate(env.snake):
                rect = pygame.Rect(
                    MARGIN + x * CELL_SIZE + 3,
                    board_origin_y + y * CELL_SIZE + 3,
                    CELL_SIZE - 6,
                    CELL_SIZE - 6,
                )
                color = SNAKE_HEAD if i == 0 else SNAKE_BODY
                pygame.draw.rect(self.screen, color, rect, border_radius=2)

        footer = self.font_small.render("Space / click to advance in step-by-step mode", True, SUBTEXT)
        self.screen.blit(footer, (MARGIN, self.window_height - 26))

    def _draw_apple(self, x: int, y: int, board_origin_y: int, color: tuple[int, int, int]) -> None:
        rect = pygame.Rect(
            MARGIN + x * CELL_SIZE + 10,
            board_origin_y + y * CELL_SIZE + 10,
            CELL_SIZE - 20,
            CELL_SIZE - 20,
        )
        pygame.draw.ellipse(self.screen, color, rect)
