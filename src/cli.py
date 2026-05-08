from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.agent import AgentConfig, QLearningAgent
from src.config import EPSILON_START
from src.environment import SnakeEnvironment
from src.trainer import Trainer
from src.gui_menu import show_menu


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser for CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Learn2Slither reinforcement learning Snake"
    )
    parser.add_argument(
        "-s", "--sessions", type=int, default=1, dest="episodes",
        help="Number of training sessions to execute"
    )
    parser.add_argument(
        "-sp", "--speed", default="human",
        help="Render speed: slow, human, normal, fast, or delay in ms"
    )
    parser.add_argument(
        "-st", "--step-by-step", action="store_true", dest="step",
        help="Enable step-by-step mode"
    )
    parser.add_argument(
        "-v", "--visual", choices=("on", "off"), default="off",
        help="Enable/disable graphical display"
    )
    parser.add_argument(
        "-nr", "--no-render", action="store_true",
        help="Disable graphical rendering"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Reduce terminal output"
    )
    parser.add_argument(
        "-l", "--load", type=Path, dest="load_model",
        help="Load a saved Q-learning model"
    )
    parser.add_argument(
        "-S", "--save", type=Path, dest="save_model",
        help="Save the Q-learning model after running"
    )
    parser.add_argument(
        "-d", "--dontlearn", "--no-learning", action="store_true",
        dest="no_learning",
        help="Run without updating the Q-table"
    )
    parser.add_argument(
        "-e", "--epsilon", type=float, default=None,
        help="Exploration rate for action selection"
    )
    parser.add_argument(
        "-hr", "--heuristics", action="store_true",
        help="Enable heuristic-guided exploration (bonus mode)"
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=None,
        help="Override the Q-learning rate"
    )
    parser.add_argument(
        "-df", "--discount-factor", type=float, default=None,
        help="Override the Q-learning discount factor"
    )
    parser.add_argument(
        "-em", "--epsilon-min", type=float, default=None,
        help="Override the minimum exploration rate"
    )
    parser.add_argument(
        "-ed", "--epsilon-decay", type=float, default=None,
        help="Override the exploration decay"
    )
    parser.add_argument(
        "-b", "--board-size", type=int, default=None,
        help="Set the board size (default: 10)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.

    Returns:
        Exit code (0 for success).
    """
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "bonus":
        argv = show_menu()

    parser = build_parser()
    args = parser.parse_args(argv)

    epsilon_value = EPSILON_START if args.epsilon is None else args.epsilon

    config = AgentConfig(
        epsilon=epsilon_value,
        learning_enabled=not args.no_learning,
        heuristics_enabled=args.heuristics,
    )
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.discount_factor is not None:
        config.discount_factor = args.discount_factor
    if args.epsilon_min is not None:
        config.epsilon_min = args.epsilon_min
    if args.epsilon_decay is not None:
        config.epsilon_decay = args.epsilon_decay

    agent = QLearningAgent(config=config)
    if args.load_model is not None:
        agent.load(args.load_model)
        if args.epsilon is not None:
            agent.config.epsilon = args.epsilon
        elif args.no_learning:
            agent.config.epsilon = 0.0
        agent.config.learning_enabled = not args.no_learning
        if args.heuristics:
            agent.config.heuristics_enabled = True
        # Reapply argument overrides after loading to ensure they take precedence
        if args.learning_rate is not None:
            agent.config.learning_rate = args.learning_rate
        if args.discount_factor is not None:
            agent.config.discount_factor = args.discount_factor
        if args.epsilon_min is not None:
            agent.config.epsilon_min = args.epsilon_min
        if args.epsilon_decay is not None:
            agent.config.epsilon_decay = args.epsilon_decay

    env = SnakeEnvironment(heuristics_enabled=agent.config.heuristics_enabled, board_size=args.board_size)
    render_enabled = args.visual == "on" and not args.no_render
    trainer = Trainer(
        env=env,
        agent=agent,
        render=render_enabled,
        step_mode=args.step,
        speed=args.speed,
        quiet=args.quiet,
    )

    result = trainer.run(
        episodes=max(args.episodes, 1),
        learn=not args.no_learning,
        epsilon_explore=agent.config.epsilon > 0,
    )

    if args.save_model is not None:
        agent.save(args.save_model)

    print(
        f"Finished {result.episodes} episodes | victories={result.victories} | "
        f"average_score={result.average_score:.2f} | best_score={result.best_score}"
    )

    return 0
