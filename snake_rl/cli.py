from __future__ import annotations

import argparse
from pathlib import Path

from snake_rl.agent import AgentConfig, QLearningAgent
from snake_rl.config import EPSILON_START
from snake_rl.environment import SnakeEnvironment
from snake_rl.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learn2Slither reinforcement learning Snake")
    parser.add_argument("-sessions", "--episodes", type=int, default=1, help="Number of training sessions to execute")
    parser.add_argument("-speed", "--speed", default="human", help="Render speed preset: slow, human, normal, fast or a delay in ms")
    parser.add_argument("-step-by-step", "--step", action="store_true", help="Enable step-by-step mode")
    parser.add_argument("-visual", "--visual", choices=("on", "off"), default="off", help="Turn the graphical display on or off")
    parser.add_argument("--no-render", action="store_true", help="Disable graphical rendering")
    parser.add_argument("-quiet", "--quiet", action="store_true", help="Reduce terminal output")
    parser.add_argument("-load", "--load-model", type=Path, help="Load a saved Q-learning model")
    parser.add_argument("-save", "--save-model", type=Path, help="Save the Q-learning model after running")
    parser.add_argument("-dontlearn", "--no-learning", action="store_true", help="Run without updating the Q-table")
    parser.add_argument("-epsilon", "--epsilon", type=float, default=None, help="Exploration rate used for action selection")
    parser.add_argument("-heuristics", "--heuristics", action="store_true", help="Enable heuristic-guided exploration (bonus mode)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override the Q-learning rate")
    parser.add_argument("--discount-factor", type=float, default=None, help="Override the Q-learning discount factor")
    parser.add_argument("--epsilon-min", type=float, default=None, help="Override the minimum exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=None, help="Override the exploration decay")
    parser.add_argument("-board-size", "--board-size", type=int, default=None, help="Set the board size (default: 10)")
    return parser


def main(argv: list[str] | None = None) -> int:
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

    if not args.quiet:
        print(
            f"Finished {result.episodes} episodes | victories={result.victories} | "
            f"average_score={result.average_score:.2f} | best_score={result.best_score}"
        )

    return 0
