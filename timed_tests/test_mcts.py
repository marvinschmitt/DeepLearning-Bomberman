"""This module tests the mcts implementation"""
import unittest
import time
import queue
import numpy as np
from MCTS.mcts import MCTS
from MCTS.bomberman_node import BombermanNode
from agents_fast import Agent
from environment_fast import BombeRLeWorld
import matplotlib.pyplot as plt

class TestMCTS(unittest.TestCase):
    time_to_think = 6e9 # time in ns

    def test_mcts(self):
        for _ in range(1):
            mcts = MCTS(C=1)
            agents = [Agent(train=True), Agent(), Agent(), Agent()]
            initial_state = BombeRLeWorld(agents)
            initial_order = queue.deque([0] + list(np.random.permutation([1, 2, 3])))
            root = BombermanNode(initial_state, np.zeros(len(agents)), initial_order, "")
            iterations = 0
            start = time.perf_counter_ns()
            while time.perf_counter_ns() - start < self.time_to_think:
                mcts.do_rollout(root)
                iterations += 1

            print(mcts.choose(root).get_action())
            print(iterations)

    def test_mcts_time(self):
        def count_mcts_iterations(time_to_think):
            mcts = MCTS(C=1000)
            agents = [Agent(train=True), Agent(), Agent(), Agent()]
            initial_state = BombeRLeWorld(agents)
            initial_order = queue.deque([0] + list(np.random.permutation([1, 2, 3])))
            root = BombermanNode(initial_state, np.zeros(len(agents)), initial_order, "")
            n_iter = 0
            start = time.perf_counter_ns()
            while time.perf_counter_ns() - start < (time_to_think*1e6):
                mcts.do_rollout(root)
                n_iter += 1
            return n_iter

        t = [i for i in range(100, 1000, 100)]+[i for i in range(1000, 10000, 500)]
        iterations = np.zeros(len(t))
        for i in range(len(t)):
            iterations[i] = count_mcts_iterations(t[i])

        plt.plot(t, iterations)
        plt.title("MCTS rollouts by think time")
        plt.xlabel("Think time [ms]")
        plt.ylabel("Number of rollouts")
        plt.savefig("mcts_rollouts_by_t.png")
        # add line at 500ms if executed on RYZEN
        # add number of rollouts in DQN and value
