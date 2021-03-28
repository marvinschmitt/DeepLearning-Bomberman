import argparse
import time
import queue
import numpy as np
import tensorflow as tf
import settings as s
from tqdm import tqdm
from MCTS.mcts import MCTS
from MCTS.bomberman_node import BombermanNode
from agents_fast import Agent
from environment_fast import BombeRLeWorld, ACTION_TO_ID

def main(args):
    start_generation = time.perf_counter_ns()

    state_queue = []
    observation = []
    value = []
    q = []

    number_of_samples = 0
    with tqdm(total=args.samples if args.samples else float("inf")) as t:
        while True:
            if args.samples and number_of_samples >= args.samples:
                break
            if args.time and time.perf_counter_ns() - start_generation > args.time:
                break

            sample = []
            mcts = MCTS(C=args.C)

            if state_queue:
                initial_state = state_queue.pop()
            else:
                agents = [Agent(train=True), Agent(), Agent(), Agent()]
                initial_state = BombeRLeWorld(agents)
                collect_states = True

            if initial_state is None:
                breakpoint()

            initial_order = queue.deque([0] + list(np.random.permutation(range(1, len(initial_state.agents)))))
            root = BombermanNode(initial_state, np.zeros(s.MAX_AGENTS), initial_order, "")

            observation.append(initial_state.get_observation())

            if len(state_queue) >= args.state_queue_length:
                collect_states = False

            states_in_this_round = []
            start = time.perf_counter_ns()
            while time.perf_counter_ns() - start < args.time_to_think:
                collect_state = len(states_in_this_round) <= 25
                intermediary_state = mcts.do_rollout(root, collect_state=collect_state)
                if collect_states and collect_state and intermediary_state:
                    states_in_this_round.append(intermediary_state)

            state_queue += states_in_this_round

            value.append(mcts.Q[root][0]/mcts.N[root])

            q_values = np.empty((6,), dtype=np.float32)
            q_values[:] = -np.inf
            for child in mcts.children[root]:
                q_values[ACTION_TO_ID[child.get_action()]] = mcts.Q[child][0]/mcts.N[child]
            q.append(q_values)

            number_of_samples += 1
            t.update()

    np.save("observation", np.array(observation))
    np.save("value", np.array(value))
    np.save("q", np.array(q))

def parse_arguments():
    """Parses the commandline arguments.

    Returns:
        Arguments extracted from the commandline.

    """
    parser = argparse.ArgumentParser(description='Generate Data from MCTS')
    parser.add_argument('--time_to_think', type=int, default=10e9,
                        help='Time to think for MCTS')
    parser.add_argument('--C', type=float, default=40,
                        help='Exploration Hyperparameter for UCT')
    parser.add_argument('--time', type=int, default=0,
                        help='Time after which to stop generation')
    parser.add_argument('--samples', type=int, default=0,
                        help='Number of samples to generate')
    parser.add_argument('--state_queue_length', type=int, default=50,
                        help='How many state are buffered for later evaluation')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(parse_arguments())
