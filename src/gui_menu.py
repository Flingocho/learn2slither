"""Graphical menu interface for Learn2Slither configuration using Pygame."""

from pathlib import Path
import pygame
from typing import Optional

from src.config import (
    BACKGROUND, PANEL_BG, TEXT, SUBTEXT, HIGHLIGHT, GREEN_APPLE
)


class Button:
    """Simple clickable button."""

    def __init__(self, x: int, y: int, width: int, height: int, text: str,
                 color: tuple[int, int, int] = HIGHLIGHT):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hovered = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw button on surface."""
        bg_color = tuple(min(c + 20, 255) for c in self.color) if self.hovered else self.color
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=4)
        pygame.draw.rect(surface, TEXT, self.rect, width=2, border_radius=4)

        text_color = (255, 255, 255) if self.hovered else (0, 0, 0)
        text_surf = font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_clicked(self, pos: tuple[int, int]) -> bool:
        """Check if button was clicked at position."""
        return self.rect.collidepoint(pos)

    def update_hover(self, pos: tuple[int, int]) -> None:
        """Update hover state."""
        self.hovered = self.rect.collidepoint(pos)


class Checkbox:
    """Simple checkbox with label."""

    def __init__(self, x: int, y: int, label: str, checked: bool = False):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.label = label
        self.checked = checked
        self.label_rect = pygame.Rect(x + 30, y, 200, 20)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw checkbox on surface."""
        pygame.draw.rect(surface, TEXT, self.rect, width=2, border_radius=2)
        if self.checked:
            pygame.draw.rect(surface, GREEN_APPLE, self.rect.inflate(-4, -4), border_radius=1)

        label_surf = font.render(self.label, True, TEXT)
        surface.blit(label_surf, self.label_rect)

    def is_clicked(self, pos: tuple[int, int]) -> bool:
        """Check if checkbox was clicked."""
        if self.rect.collidepoint(pos):
            self.checked = not self.checked
            return True
        return False

    def toggle(self) -> None:
        """Toggle checkbox state."""
        self.checked = not self.checked


class TextInput:
    """Simple text input field."""

    def __init__(self, x: int, y: int, width: int, label: str, default: str = ""):
        self.rect = pygame.Rect(x, y, width, 35)
        self.label = label
        self.text = default
        self.active = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw text input on surface."""
        label_surf = font.render(self.label, True, SUBTEXT)
        surface.blit(label_surf, (self.rect.x, self.rect.y - 25))

        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=4)
        border_color = HIGHLIGHT if self.active else TEXT
        pygame.draw.rect(surface, border_color, self.rect, width=2, border_radius=4)

        text_surf = font.render(self.text, True, TEXT)
        surface.blit(text_surf, (self.rect.x + 8, self.rect.y + 8))

    def is_clicked(self, pos: tuple[int, int]) -> bool:
        """Check if input field was clicked."""
        if self.rect.collidepoint(pos):
            self.active = True
            return True
        self.active = False
        return False

    def handle_key(self, event: pygame.event.Event) -> None:
        """Handle keyboard input."""
        if not self.active:
            return

        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        elif event.key == pygame.K_RETURN:
            self.active = False
        elif len(self.text) < 10:
            if event.unicode.isdigit():
                self.text += event.unicode


class ModelSelector:
    """Scrollable model selector."""

    def __init__(self, x: int, y: int, width: int, height: int, models: list[Path]):
        self.rect = pygame.Rect(x, y, width, height)
        self.models = [m.name for m in models]
        self.scroll_offset = 0
        self.selected_idx = 0
        self.item_height = 30

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw model selector on surface."""
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=4)
        pygame.draw.rect(surface, TEXT, self.rect, width=2, border_radius=4)

        visible_items = self.rect.height // self.item_height
        start = self.scroll_offset
        end = min(start + visible_items, len(self.models))

        for i, idx in enumerate(range(start, end)):
            item_rect = pygame.Rect(
                self.rect.x + 5,
                self.rect.y + i * self.item_height + 5,
                self.rect.width - 10,
                self.item_height - 5
            )

            if idx == self.selected_idx:
                pygame.draw.rect(surface, HIGHLIGHT, item_rect, border_radius=2)
                text_color = BACKGROUND
            else:
                text_color = TEXT

            text_surf = font.render(self.models[idx], True, text_color)
            text_rect = text_surf.get_rect(midleft=(item_rect.x + 5, item_rect.centery))
            surface.blit(text_surf, text_rect)

    def is_clicked(self, pos: tuple[int, int]) -> Optional[int]:
        """Get selected model index if clicked."""
        if not self.rect.collidepoint(pos):
            return None

        visible_items = self.rect.height // self.item_height
        click_y = pos[1] - self.rect.y
        item_idx = click_y // self.item_height

        if item_idx < visible_items:
            actual_idx = self.scroll_offset + item_idx
            if actual_idx < len(self.models):
                self.selected_idx = actual_idx
                return actual_idx
        return None


def get_available_models() -> list[Path]:
    """Get list of available model files."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.json"))


def show_menu() -> list[str]:
    """Show graphical menu and return CLI arguments."""
    pygame.init()
    window_width, window_height = 900, 700
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Learn2Slither - Configuration Menu")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    title_font = pygame.font.Font(None, 48)
    label_font = pygame.font.Font(None, 22)

    models = get_available_models()
    selector = ModelSelector(50, 80, 300, 200, models)
    selected_model: Optional[int] = None if models else None

    options = {
        "episodes": "1000",
        "visualization": False,
        "speed": "human",
        "step_by_step": False,
        "quiet": False,
        "heuristics": False,
        "epsilon": "",
        "learning_rate": "",
        "discount_factor": "",
        "epsilon_decay": "",
        "board_size": "",
        "save_model": False,
        "no_learn": False,
    }

    episodes_input = TextInput(400, 330, 300, "Number of sessions:", "1000")
    board_size_input = TextInput(400, 420, 300, "Board size (NxN):", "10")

    checkboxes = [
        ("visualization", Checkbox(400, 100, "Enable visualization", False)),
        ("step_by_step", Checkbox(400, 135, "Step-by-step mode", False)),
        ("quiet", Checkbox(400, 170, "Quiet mode", False)),
        ("heuristics", Checkbox(400, 205, "Heuristics enabled", False)),
        ("save_model", Checkbox(400, 240, "Save model", False)),
        ("no_learn", Checkbox(400, 275, "Don't learn (eval only)", False)),
    ]

    buttons = {
        "start": Button(350, 600, 200, 40, "Start Training"),
        "cancel": Button(575, 600, 100, 40, "Cancel"),
    }

    running = True
    result = None
    mouse_pos = pygame.mouse.get_pos()

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                result = None

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                episodes_input.is_clicked(mouse_pos)
                board_size_input.is_clicked(mouse_pos)

                for key, checkbox in checkboxes:
                    if checkbox.is_clicked(mouse_pos):
                        options[key] = checkbox.checked

                if selector.is_clicked(mouse_pos) is not None:
                    selected_model = selector.selected_idx

                if buttons["start"].is_clicked(mouse_pos):
                    if episodes_input.text:
                        options["episodes"] = episodes_input.text
                    if board_size_input.text:
                        options["board_size"] = board_size_input.text
                    result = build_args(options, models, selected_model)
                    running = False

                if buttons["cancel"].is_clicked(mouse_pos):
                    running = False
                    result = None

            if event.type == pygame.KEYDOWN:
                episodes_input.handle_key(event)
                board_size_input.handle_key(event)

            if event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
                buttons["start"].update_hover(mouse_pos)
                buttons["cancel"].update_hover(mouse_pos)

        screen.fill(BACKGROUND)

        title = title_font.render("Learn2Slither - Configuration", True, TEXT)
        screen.blit(title, (30, 20))

        models_label = label_font.render("Available Models:", True, SUBTEXT)
        screen.blit(models_label, (50, 60))

        selector.draw(screen, font)

        pygame.draw.line(screen, PANEL_BG, (370, 50), (370, 550), width=1)

        options_label = label_font.render("Options:", True, SUBTEXT)
        screen.blit(options_label, (400, 55))

        for _, checkbox in checkboxes:
            checkbox.draw(screen, font)

        episodes_input.draw(screen, font)
        board_size_input.draw(screen, font)

        buttons["start"].draw(screen, font)
        buttons["cancel"].draw(screen, font)

        pygame.display.flip()

    pygame.quit()
    return result or []


def build_args(options: dict, models: list[Path], selected_model: Optional[int]) -> list[str]:
    """Build CLI arguments from menu selections."""
    args = []

    if selected_model is not None and selected_model < len(models):
        args.extend(["-l", f"models/{models[selected_model].name}"])
        if options["no_learn"]:
            args.append("-d")

    args.extend(["-s", options["episodes"]])
    args.extend(["-v", "on" if options["visualization"] else "off"])

    if options["visualization"]:
        args.extend(["-sp", options["speed"]])
        if options["step_by_step"]:
            args.append("-st")

    if options["quiet"]:
        args.append("-q")

    if options["heuristics"]:
        args.append("-hr")

    if options["epsilon"]:
        args.extend(["-e", options["epsilon"]])

    if options["learning_rate"]:
        args.extend(["-lr", options["learning_rate"]])

    if options["discount_factor"]:
        args.extend(["-df", options["discount_factor"]])

    if options["epsilon_decay"]:
        args.extend(["-ed", options["epsilon_decay"]])

    if options["board_size"]:
        args.extend(["-b", options["board_size"]])

    if options["save_model"]:
        args.extend(["-S", "models/qtable_trained.json"])

    return args
