"""rlm — Reinforcement Learning with Language Models.

A library for training and evaluating language models using reinforcement
learning techniques, including RLHF, PPO, and reward modeling.

Personal fork notes:
- Using this for experimenting with custom reward shaping ideas
- See examples/ for my personal training scripts
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rlm")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
