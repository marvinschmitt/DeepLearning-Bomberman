"""This module tests the mcts implementation"""
import unittest
import time
import queue
import numpy as np
from MCTS.mcts import MCTS
from MCTS.bomberman_node import BombermanNode
from agents_fast import Agent
from environment_fast import BombeRLeWorld

class TestMCTS(unittest.TestCase):
    time_to_think = 6e9 # time in ns

    def test_mcts(self):
        for _ in range(20):
            mcts = MCTS(C=1)
            agents = [Agent(True), Agent(), Agent(), Agent()]
            initial_state = BombeRLeWorld(agents)
            initial_order = queue.deque([0] + np.random.permutation([1, 2, 3]))
            root = BombermanNode(initial_state, np.zeros(len(agents)), initial_order, "")

            iterations = 0
            start = time.perf_counter_ns()
            while time.perf_counter_ns() - start < self.time_to_think:
                mcts.do_rollout(root)
                iterations += 1

            print(mcts.choose(root).get_action())
            print(iterations)
