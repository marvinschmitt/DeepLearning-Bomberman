from mcts import Node
from typing import Tuple

class BombermanNode(Node):
    """
    A representation of a single state of the Bomberman game.
    Needs to be able to direct all player actions.

    This is basically the actual board state + the agent who acts next + the action that was taken to get here.

    Implementation Hint: Maybe call 'perform_agent_action' from 'GenericWorld' directly?
    """
    def __init__(self, actor, action):
        self.actor = actor
        self.action = action

    def get_actor(self) -> int:
        "Return who is taking the action. (index in 0,..,3)"
        pass

    def get_action(self) -> str:
        "Return who is taking the action. (index in 0,..,3)"
        pass

    def find_children(self) -> set(Node):
        "All possible successors of this board state."
        pass

    def find_child_for_rollout(self) -> Tuple[Node, float]:
        "Successor of this board state (for now just a randomly selected action)+ reward"
        pass

    def is_terminal(self) -> bool:
        "Returns True if the node has no children."
        pass

    def __hash__(self) -> int:
        "Nodes must be hashable"
        pass

    def __eq__(node1, node2) -> bool:
        "Nodes must be comparable"
        pass