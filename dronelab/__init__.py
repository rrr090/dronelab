"""dronelab: reinforcement learning for drone teams carrying slung cargo.

Subpackages:
    hover    planar single-drone hover task (the starting point)
    cargo    three 3D quadrotors holding one cargo on elastic ropes
    mission  the cargo team flying through procedurally generated maps

Shared:
    policy   actor-critic MLP
    ppo      vectorised PPO trainer
    viz      helpers for the three.js HTML viewers
"""

__version__ = "0.1.0"
