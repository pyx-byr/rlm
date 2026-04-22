"""Core module for rlm — Reinforcement Learning with Language Models.

This module provides the fundamental building blocks for training and
evaluating language models using reinforcement learning techniques.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RLMConfig:
    """Configuration for an RLM training run.

    Attributes:
        model_name: Identifier for the base language model.
        learning_rate: Learning rate for the policy optimizer.
        gamma: Discount factor for future rewards.
        clip_epsilon: PPO clipping parameter.
        max_steps: Maximum number of training steps.
        batch_size: Number of episodes per training batch.
        reward_fn: Optional custom reward function.
        device: Compute device (e.g. 'cpu', 'cuda').
    """

    model_name: str = "gpt2"
    learning_rate: float = 1e-5
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    max_steps: int = 10_000
    batch_size: int = 32
    reward_fn: Optional[Callable[[str, str], float]] = None
    device: str = "cpu"
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration values and raise on invalid settings."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not (0.0 < self.clip_epsilon < 1.0):
            raise ValueError(f"clip_epsilon must be in (0, 1), got {self.clip_epsilon}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class Episode:
    """A single rollout episode collected during environment interaction.

    Attributes:
        prompt: The input prompt provided to the language model.
        response: The generated text response.
        reward: Scalar reward signal for this episode.
        log_prob: Log-probability of the response under the policy.
        metadata: Optional auxiliary data attached to the episode.
    """

    prompt: str
    response: str
    reward: float
    log_prob: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RLMTrainer:
    """High-level trainer that orchestrates the RL fine-tuning loop.

    Example::

        config = RLMConfig(model_name="gpt2", max_steps=1000)
        trainer = RLMTrainer(config)
        trainer.train(prompts=["Explain gravity in one sentence."])
    """

    def __init__(self, config: RLMConfig) -> None:
        config.validate()
        self.config = config
        self._step: int = 0
        self._episode_buffer: List[Episode] = []
        logger.info(
            "Initialized RLMTrainer with model '%s' on device '%s'.",
            config.model_name,
            config.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_episode(self, prompt: str, response: str, reward: float, log_prob: float) -> Episode:
        """Record a completed episode and add it to the replay buffer."""
        ep = Episode(prompt=prompt, response=response, reward=reward, log_prob=log_prob)
        self._episode_buffer.append(ep)
        return ep

    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted cumulative returns for a sequence of rewards."""
        returns: List[float] = []
        running = 0.0
        for r in reversed(rewards):
            running = r + self.config.gamma * running
            returns.insert(0, running)
        return returns

    def step(self) -> Optional[Tuple[List[Episode], List[float]]]:
        """Consume the episode buffer and return a batch with computed returns.

        Returns ``None`` when the buffer has fewer episodes than ``batch_size``.
        """
        if len(self._episode_buffer) < self.config.batch_size:
            return None

        batch = self._episode_buffer[: self.config.batch_size]
        self._episode_buffer = self._episode_buffer[self.config.batch_size :]
        rewards = [ep.reward for ep in batch]
        returns = self.compute_returns(rewards)
        self._step += 1
        logger.debug("Training step %d completed with batch size %d.", self._step, len(batch))
        return batch, returns

    @property
    def global_step(self) -> int:
        """Number of completed training steps."""
        return self._step
