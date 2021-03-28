"""
This Module provides an implementation of the MCTS algorithm. To use it one has to implement the
abstract Node class.
"""
from __future__ import annotations
import math
import random
from copy import deepcopy

import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple



class MCTS:
    "Monte Carlo tree Searcher. Multiplayer reward for 4 players."

    def __init__(self, C=1, path_selection_policy=None, final_selection_policy=None):
        self.Q = defaultdict(lambda: np.array([0., 0., 0., 0.]))  # total reward of each node (four players)
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.C = C  # Exploration hyperparameter in UTCs
        self.final_selection_policy = final_selection_policy # policy used to select action
        self.path_selection_policy = path_selection_policy or self._uct_select

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_child_for_rollout()

        def score(n):
            assert node.get_actor() == 0
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n][node.get_actor()] / self.N[n]  # average reward

        selection_policy = self.final_selection_policy or score

        print([self.Q[child]/self.N[child] for child in self.children[node]])
        return max(self.children[node], key=selection_policy)

    def do_rollout(self, node, collect_state=False):
        "Perform a single rollout. Return one of the intermediary states if collect_state."
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward, state = self._simulate(leaf, collect_state=collect_state)
        self._backpropagate(path, reward)

        return state

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path

            assert isinstance(self.children[node], set)
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = random.choice(list(unexplored))
                path.append(n)
                return path
            node = self.path_selection_policy(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node, collect_state=False):
        "Returns the reward for a random simulation (to completion) of `node`. Return intermediary state if collect_state."
        collected_state = None

        while not node.is_terminal():
            node = node.find_child_for_rollout()
            if node.get_actor() == 0 and collect_state:
                if random.random() < np.exp(-node.world.step):
                    collected_state = deepcopy(node.world)

        return node.get_reward(), collected_state

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        # Mustn't be terminal
        assert not node.is_terminal()
        if node.get_actor() is None:
            i = 1

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"

            return self.Q[n][node.get_actor()] / self.N[n] + self.C * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def get_actor(self) -> int:
        "Return who is taking the action. (index in 0,..,3)"
        pass

    @abstractmethod
    def get_action(self) -> str:
        "Return what action was taken to get to this state."
        pass

    @abstractmethod
    def get_reward(self) -> np.array:
        "Return what action was taken to get to this state."
        pass

    @abstractmethod
    def find_children(self) -> set[Node]:
        "All possible successors of this board state."
        pass

    @abstractmethod
    def find_child_for_rollout(self) -> Tuple[Node, float]:
        "Successor of this board state (for now just a randomly selected action)+ reward"
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        "Returns True if the node has no children."
        pass

    @abstractmethod
    def __hash__(self) -> int:
        "Nodes must be hashable"
        pass

    @abstractmethod
    def __eq__(node1, node2) -> bool:
        "Nodes must be comparable"
        pass
