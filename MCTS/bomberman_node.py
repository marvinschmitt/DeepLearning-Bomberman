from __future__ import annotations
import random
from MCTS.mcts import Node
from typing import Tuple, List
from copy import deepcopy

import queue

import numpy as np
from collections import namedtuple

from environment_fast import BombeRLeWorld, ACTIONS



class BombermanNode(Node):
    """
    A representation of a single state of the Bomberman game.
    Needs to be able to direct all player actions.

    This is basically the actual board state + the agent who acts next + the action that was taken to get here.
    """
    def __init__(self, world: BombeRLeWorld, reward: np.array, actor_queue: queue.deque, action: str, rollout=False):
        self.world = world

        if not actor_queue:
            self.world.do_step()
            active_agent_indices = [a for a in range(len(self.world.agents)) if self.world.agents[a] in self.world.active_agents]
            actor_queue = queue.deque(np.random.permutation(active_agent_indices))
        self.actor_queue = actor_queue

        self.actor = self.actor_queue.popleft() if self.actor_queue else None # (terminal state)
        self.action = action

        self.reward = deepcopy(reward)
        for a in range(len(self.world.agents)):
            reward[a] += self.world.agents[a].process_reward_delta()

        self.rollout = rollout # If node is in rollout mode it won't be deepcopied when an action is performed.

    def get_actor(self) -> int:
        "Return who is taking the action. (index in 0,..,3)"
        return self.actor

    def get_action(self) -> str:
        "Return what action was taken. (index in 0,..,5)"
        return self.action

    def get_reward(self) -> np.array:
        "Return what action was taken to get to this state."
        return self.reward

    def find_children(self) -> List[Node]:
        "All possible successors of this board state."
        if self.is_terminal():
            return []

        valid_actions = [action for action in ACTIONS if self.is_action_valid(action)]

        return [self.make_move(self.actor, action) for action in valid_actions]

    def find_child_for_rollout(self) -> Node:
        "Successor of this board state (for now just a randomly selected action)+ reward"
        valid_actions = [action for action in ACTIONS if self.is_action_valid(action)]
        action = random.choice(valid_actions)
        node = self.make_move(self.actor, action, rollout=True)
        return node

    def is_terminal(self) -> bool:
        "Returns True if the node has no children."
        return not self.world.running

    def __hash__(self) -> int:
        "Nodes must be hashable"
        return id(self)

    def __eq__(self, node2) -> bool:
        "Nodes must be comparable"
        if self.world != node2.world:
            return False

        if self.actor_queue != node2.actor_queue:
            return False

        if self.actor != node2.actor:
            return False

        if self.action != node2.action:
            return False

        return True

    def make_move(self, agent_idx, action, rollout=False):
        world = deepcopy(self.world) # self.world if self.rollout else
        a = world.agents[agent_idx]
        world.perform_agent_action(a, action)
        node = BombermanNode(world, self.reward, self.actor_queue, action, rollout)
        return node

    def is_action_valid(self, action):
        agent = self.world.agents[self.actor]
        return self.world.is_action_valid(agent, action)
