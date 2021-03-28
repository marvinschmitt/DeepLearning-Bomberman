import time
import queue
import numpy as np
import settings as s
from .mcts import MCTS
from .bomberman_deep_node import BombermanDeepNode
from .environment_fast import BombeRLeWorld
from collections import defaultdict
from tensorflow.keras.models import load_model

class Predictor:
    def __init__(self, dqn_path):
        self.q_net = load_model(dqn_path)

    def predict(self, observation):
        return np.mean(self.q_net(observation[np.newaxis, :]))

def setup(self):
    self.time_to_think = 450e6 # in ms
    self.C = 40 # Exploration Hyperparameter
    self.predictor = Predictor("best-network.hdf5")

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
    start = time.perf_counter_ns()
    if game_state["step"] == 1:
        init(self)

    log_bombs(self, game_state)
    log_coins(self, game_state)

    mcts = MCTS(C=self.C)
    initial_state = BombeRLeWorld(game_state, self.bomb_log, self.coin_log)
    initial_order = queue.deque([0] + list(np.random.permutation(range(1, len(initial_state.agents)))))
    root = BombermanDeepNode(initial_state, self.predictor, initial_order, "")
    if self.first_turn:
        root.suppress_bomb()
        self.first_turn = False

    iterations = 0
    while time.perf_counter_ns() - start < self.time_to_think:
        mcts.do_rollout(root)
        iterations += 1

    action = mcts.choose(root).get_action()

    return action