"""
Adapted from
Luke Harold Miles, July 2019, Public Domain Dedication
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from __future__ import annotations

import math
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple



class MCTS:
    "Monte Carlo tree Searcher. Multiplayer reward for 4 players."

    def __init__(self, C=1, path_selection_policy=None, final_selection_policy=None):
        self.Q = defaultdict(lambda: np.array([0, 0, 0, 0]))  # total reward of each node (four players)
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
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n][n.get_actor()] / self.N[n]  # average reward

        selection_policy = self.final_selection_policy or score

        return max(self.children[node], key=selection_policy)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self.path_selection_policy(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        reward = np.array([0, 0, 0, 0])
        while not node.is_terminal():
            node, reward = node.find_child_for_rollout()
            reward += reward

        return reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n][n.get_actor()] / self.N[n] + self.C * math.sqrt(
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