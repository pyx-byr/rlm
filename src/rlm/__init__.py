"""rlm — Reinforcement Learning with Language Models.

A library for training and evaluating language models using reinforcement
learning techniques, including RLHF, PPO, and reward modeling.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rlm")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
