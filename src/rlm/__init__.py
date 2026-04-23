"""rlm — Reinforcement Learning with Language Models.

A library for training and evaluating language models using reinforcement
learning techniques, including RLHF, PPO, and reward modeling.

Personal fork notes:
- Using this for experimenting with custom reward shaping ideas
- See examples/ for my personal training scripts
- Added DEFAULT_KL_COEF for quick tuning without digging into configs
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rlm")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Personal default: upstream uses 0.2, but I found 0.05 works better
# for my smaller-scale experiments where KL blows up quickly.
DEFAULT_KL_COEF = 0.05

__all__ = ["__version__", "DEFAULT_KL_COEF"]
