import logging

from fallbacks import pygame

# Game properties
COLS = 9 # min 3
ROWS = 9 # min 3
CRATE_DENSITY = 0.10
MAX_AGENTS = 1

# Round properties
MAX_STEPS = 100

# GUI properties
GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2

AGENT_COLORS = ['blue', 'green', 'yellow', 'pink'] * 100

# Game rules
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2

# Rules for agents
TIMEOUT = 0.5
REWARD_KILL = 5
REWARD_COIN = 1

# User input
INPUT_MAP = {
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    pygame.K_RETURN: 'WAIT',
    pygame.K_SPACE: 'BOMB',
}

# Logging levels
LOG_GAME = logging.ERROR
LOG_AGENT_WRAPPER = logging.ERROR
LOG_AGENT_CODE = logging.ERROR
LOG_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
