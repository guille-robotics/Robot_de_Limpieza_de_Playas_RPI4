"""
robot_env.py
Registra el entorno RobotSimulacion‑v0 para usarlo con gym.make().
Se ejecuta una sola vez, cuando el módulo se importa.
"""

import gymnasium as gym
from gymnasium.envs.registration import register

# ── REGISTRO SEGURO ────────────────────────────────────────────
# Gym lanza un Error si se intenta registrar el mismo id dos veces
try:
    register(
        id="RobotReal-v0",
        entry_point="ambiente_robot_real:RobotReal",  # <nombre_del_codigo>:<clase>
        max_episode_steps=None      # dejamos que el propio env trunque
    )
except gym.error.Error:
    # Ya estaba registrado en esta sesión; lo ignoramos
    pass

