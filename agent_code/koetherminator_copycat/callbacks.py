import time
import queue
import numpy as np
import settings as s
from .mcts import MCTS
from .bomberman_deep_node import BombermanDeepNode
from .environment_fast import BombeRLeWorld
from collections import defaultdict
from tensorflow.keras.models import load_model

import pickle

from agent_code.koetherminator_copycat import chicken


class ChickenPredictor:
    def __init__(self, dqn_path):
        with open(dqn_path, "rb") as file:
            self.Q = pickle.load(file)

    def predict(self, game_state):
        feat = chicken.state_to_features(game_state)
        Qs = self.Q[tuple(feat)]
        return np.max(Qs)


def setup(self):
    self.time_to_think = 450e6 # in ms
    self.C = 40 # Exploration Hyperparameter
    self.predictor = ChickenPredictor("chicken_model.pt")


def init(self):
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
    start = time.perf_counter_ns()
    if game_state["step"] == 1:
        init(self)

    log_bombs(self, game_state)
    log_coins(self, game_state)

    mcts = MCTS(C=self.C)
    initial_state = BombeRLeWorld(game_state, self.bomb_log, self.coin_log)
    initial_order = queue.deque([0] + list(np.random.permutation(range(1, len(initial_state.agents)))))
    root = BombermanDeepNode(initial_state, self.predictor, initial_order, "")


    iterations = 0
    while time.perf_counter_ns() - start < self.time_to_think:
        mcts.do_rollout(root)
        iterations += 1
    print(iterations)

    action = mcts.choose(root).get_action()

    return action