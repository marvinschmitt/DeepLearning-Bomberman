import cProfile
import pstats
import time
import queue
import numpy as np
from MCTS.mcts import MCTS
from MCTS.bomberman_node import BombermanNode
from agents_fast import Agent
from environment_fast import BombeRLeWorld


def main():
    time_to_think = 6e9
    for _ in range(10):
        mcts = MCTS(C=1000)
        agents = [Agent(train=True), Agent(), Agent(), Agent()]
        initial_state = BombeRLeWorld(agents)
        initial_order = queue.deque([0] + list(np.random.permutation([1, 2, 3])))
        root = BombermanNode(initial_state, np.zeros(len(agents)), initial_order, "")
        iterations = 0
        start = time.perf_counter_ns()
        while time.perf_counter_ns() - start < time_to_think:
            mcts.do_rollout(root)
            iterations += 1

        print(mcts.choose(root).get_action())
        print(iterations)


if __name__ == '__main__':
    cProfile.run("main()", "output.dat")

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("cumtime").print_stats()
