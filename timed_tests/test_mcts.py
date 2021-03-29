"""This module tests the mcts implementation"""
import unittest
import time
import queue
import numpy as np
from tensorflow.python.keras.models import load_model

from MCTS.mcts import MCTS
from MCTS.bomberman_node import BombermanNode
from MCTS.bomberman_deep_node import BombermanDeepNode
from agents_fast import Agent
from environment_fast import BombeRLeWorld
from agent_code.koetherminator.callbacks import Predictor
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
            mcts = MCTS(C=40)
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

        def count_mcts_iterations_q_net(time_to_think):
            predictor = Predictor("best-network.hdf5")
            mcts = MCTS(C=40)
            agents = [Agent(train=True), Agent(), Agent(), Agent()]
            initial_state = BombeRLeWorld(agents)
            initial_order = queue.deque([0] + list(np.random.permutation(range(1, len(initial_state.agents)))))
            root = BombermanDeepNode(initial_state, predictor, initial_order, "")
            root.suppress_bomb()

            n_iter = 0
            start = time.perf_counter_ns()
            while time.perf_counter_ns() - start < time_to_think:
                mcts.do_rollout(root)
                n_iter += 1

            return n_iter

        t = [i for i in range(100, 1000, 100)]+[i for i in range(1000, 5000, 1000)]
        iterations = np.zeros(len(t))
        iterations_q = np.zeros(len(t))
        for i in range(len(t)):
            #iterations[i] = count_mcts_iterations(t[i])
            iterations_q[i] = count_mcts_iterations_q_net(t[i])

        #plt.plot(t, iterations, color="blue")
        plt.plot(t, iterations_q, color="green")
        plt.title("MCTS rollouts by think time")
        plt.xlabel("Think time [ms]")
        plt.ylabel("Number of rollouts")
        plt.savefig("mcts_rollouts_by_t.png")
        # add line at 500ms if executed on RYZEN
        # add number of rollouts in DQN and value

    def test_forwardpass_time(self):
        agents = [Agent(train=True), Agent(), Agent(), Agent()]
        initial_state = BombeRLeWorld(agents)

        q_net = load_model("../results_checkpoints/pre_training/best-network.hdf5")
        observation = initial_state.get_observation()
        start = time.perf_counter_ns()
        q_net(observation[np.newaxis, :])
        print((time.perf_counter_ns() - start)/1e6)