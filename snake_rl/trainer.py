from __future__ import annotations

from dataclasses import dataclass

from snake_rl.agent import QLearningAgent
from snake_rl.config import ACTION_NAMES, SPEED_PRESETS
from snake_rl.environment import SnakeEnvironment


@dataclass
class TrainingResult:
    episodes: int
    victories: int
    average_score: float
    best_score: int


class Trainer:
    def __init__(
        self,
        env: SnakeEnvironment,
        agent: QLearningAgent,
        render: bool = True,
        step_mode: bool = False,
        speed: str | int = "human",
        quiet: bool = False,
    ) -> None:
        self.env = env
        self.agent = agent
        self.render_enabled = render
        self.step_mode = step_mode
        self.quiet = quiet
        self.speed_label, self.delay_ms = self._resolve_speed(speed)
        self.renderer = None
        self.RenderSnapshot = None
        if render:
            try:
                from snake_rl.renderer import RenderSnapshot, SnakeRenderer
            except ImportError as exc:
                raise RuntimeError(
                    "Graphical rendering requested but pygame is not installed. Use --no-render for headless training."
                ) from exc

            self.renderer = SnakeRenderer(board_size=env.board_size)
            self.RenderSnapshot = RenderSnapshot
        self.running = True

    def run(self, episodes: int, learn: bool = True, epsilon_explore: bool = True) -> TrainingResult:
        victories = 0
        total_score = 0
        best_score = 0
        checkpoint = episodes // 10 if episodes >= 10 else 1
        last_checkpoint = 0

        try:
            for episode in range(1, episodes + 1):
                state = self.env.reset()
                done = False
                step = 0

                while not done and self.running:
                    if self.renderer is not None and not self.renderer.pump_events():
                        self.running = False
                        break

                    action = self.agent.choose_action(state, explore=epsilon_explore)
                    result = self.env.step(action)

                    if learn:
                        self.agent.update(state, action, result.reward, result.state, result.done)

                    state = result.state
                    done = result.done
                    step += 1

                    if self.renderer is not None:
                        snapshot = self.RenderSnapshot(
                            episode=episode,
                            step=step,
                            score=self.env.score,
                            epsilon=self.agent.config.epsilon,
                            speed_label=self.speed_label,
                            action_name=result.info.get("action_name", ACTION_NAMES[action]),
                            reward=result.reward,
                            state=state,
                            vision=result.info.get("vision", self.env.describe_vision()),
                            reason=result.info.get("reason", ""),
                            paused=False,
                        )
                        self.renderer.render(self.env, snapshot)
                        if self.step_mode and self.running:
                            self.running = self.renderer.wait_for_step()
                        elif self.delay_ms > 0:
                            if not self.renderer.pump_events():
                                self.running = False
                                break
                            self.renderer.clock.tick(1000 // max(self.delay_ms, 1))

                    if done:
                        break

                if not self.running:
                    break

                total_score += self.env.score
                best_score = max(best_score, self.env.score)
                if self.env.score >= 10:
                    victories += 1

                if learn:
                    self.agent.decay_exploration()

                if episode - last_checkpoint >= checkpoint and checkpoint > 0:
                    avg_score = total_score / episode
                    progress_pct = (episode / episodes) * 100
                    print(f"[{progress_pct:5.1f}%] Ep {episode:5d} | Best: {best_score:2d} | Avg: {avg_score:5.2f} | Wins: {victories:3d} | ε: {self.agent.config.epsilon:.3f}")
                    last_checkpoint = episode

            average_score = total_score / max(episodes, 1)
            print("\n" + "="*70)
            print(f"TRAINING COMPLETE")
            print(f"{'Episodes:':<20} {episodes}")
            print(f"{'Best Score:':<20} {best_score}")
            print(f"{'Average Score:':<20} {average_score:.2f}")
            print(f"{'Victories (≥10):':<20} {victories}")
            print(f"{'Final ε (epsilon):':<20} {self.agent.config.epsilon:.3f}")
            print("="*70 + "\n")
            return TrainingResult(episodes=episodes, victories=victories, average_score=average_score, best_score=best_score)
        finally:
            if self.renderer is not None:
                self.renderer.close()

    def _resolve_speed(self, speed: str | int) -> tuple[str, int]:
        if isinstance(speed, int):
            return f"{speed} ms", max(speed, 0)
        if speed.isdigit():
            value = int(speed)
            return f"{value} ms", max(value, 0)
        if speed not in SPEED_PRESETS:
            speed = "human"
        return speed, SPEED_PRESETS[speed]

