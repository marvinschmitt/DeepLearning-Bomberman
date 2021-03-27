from MCTS.mcts_rewards import reward_from_events

class Agent:
    """
    The Agent game object.

    A faster, non-threaded version of the Agent meant to work with mtcs.
    """

    def __init__(self, train: bool = False):
        """The game enters a terminal state when the last agent with  train = true dies."""

        self.train = train

        self.dead = None

        self.events = None

        self.x = None
        self.y = None
        self.bombs_left = None

    def start_round(self):
        self.dead = False

        self.events = []

        self.bombs_left = True

    def add_event(self, event):
        self.events.append(event)

    def get_state(self):
        """Provide information about this agent for the global game state."""
        return self.bombs_left, (self.x, self.y)

    def reset_game_events(self):
        self.events = []

    def get_reward_delta(self):
        """Calculates reward from unprocessed events. Clears event log."""
        reward_from_events(self.events)
        self.reset_game_events()