import time
import queue
import numpy as np
import settings as s
from MCTS.mcts import MCTS
from MCTS.bomberman_node import BombermanNode
from environment_fast import BombeRLeWorld
from collections import defaultdict

def setup(self):
    self.time_to_think = 10000e6 # in ms
    self.C = 40 # Exploration Hyperparameter

def init(self):
    self.first_turn = True

    self.bomb_log = defaultdict(None)
    self.coin_log = np.zeros((s.ROWS, s.COLS))

def log_bombs(self, game_state):
    agents = dict()
    for agent in [game_state["self"]] + game_state["others"]:
        agents[agent[3]] = agent[0]

    for bomb in game_state["bombs"]:
        if bomb[0] in agents: # agent on top of bomb
            self.bomb_log[bomb[0]] = agents[bomb[0]]

def log_coins(self, game_state):
    for coin in game_state["coins"]:
        self.coin_log[coin] = 1

def act(self, game_state: dict):
    if game_state["step"] == 1:
        init(self)

    log_bombs(self, game_state)
    log_coins(self, game_state)

    mcts = MCTS(C=self.C)
    initial_state = BombeRLeWorld(game_state, self.bomb_log, self.coin_log)
    initial_order = queue.deque([0] + list(np.random.permutation(range(1, len(initial_state.agents)))))
    root = BombermanNode(initial_state, np.zeros(s.MAX_AGENTS), initial_order, "")
    if self.first_turn:
        root.suppress_bomb()
        self.first_turn = False

    start = time.perf_counter_ns()
    iterations = 0
    while time.perf_counter_ns() - start < self.time_to_think:
        mcts.do_rollout(root)
        iterations += 1

    print(iterations)
    action = mcts.choose(root).get_action()
    print(action)
    return action