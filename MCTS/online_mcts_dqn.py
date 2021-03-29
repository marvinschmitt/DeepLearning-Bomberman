import argparse
from MCTS.mcts import MCTS
from MCTS.bomberman_node import BombermanNode
from agents_fast import Agent
from environment_fast import BombeRLeWorld, ACTION_TO_ID
from MCTS.generate_data import main as generate_batch
from deep_bomber.pre_training import main as train
from collections import namedtuple


def parse_arguments():
    parser = argparse.ArgumentParser(description="MCTS online learning")

    parser.add_argument('--time_to_think', type=int, default=10e9,
                        help='Time to think for MCTS')
    parser.add_argument('--time', type=int, default=0,
                        help='Time after which to stop generation')
    parser.add_argument('--C', type=float, default=40,
                        help='Exploration Hyperparameter for UCT')
    parser.add_argument('--samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--state_queue_length', type=int, default=50,
                        help='How many state are buffered for later evaluation')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    M = 20
    args = parse_arguments()
    for i in range(M):
        observation, value, q = generate_batch(args, return_data=True, print_state_step=True)
        train(batch_size=args.samples,
              initial_epoch=i*args.samples,
              X=observation, y=q)  # extend s.th. X,y are appended to current train set


