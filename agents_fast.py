from MCTS.mcts_rewards import reward_from_events

class Agent:
    """
    The Agent game object.

    A faster, non-threaded version of the Agent meant to work with mtcs.
    """

    def __init__(self, state = None, train: bool = False):
        """The game enters a terminal state when the last agent with  train = true dies."""

        self.train = train

        self.name = "Jaqen Hghar"

        self.dead = None

        self.events = []

        self.x = None
        self.y = None
        self.bombs_left = None

        if state:
            self.name = state[0]
            self.x = state[3][0]
            self.y = state[3][1]
            self.bombs_left = state[2]

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

    def process_reward_delta(self):
        """Calculates reward from unprocessed events. Clears event log."""
        reward = reward_from_events(self.events)
        self.reset_game_events()
        return reward

    def __hash__(self) -> int:
        return hash((
                self.train,
                self.dead,
                self.x,
                self.y,
                self.bombs_left
            ))

    def __eq__(self, agent) -> bool:
        if self.train != agent.train:
            return False

        if self.dead != agent.dead:
            return False

        if self.events != agent.events:
            return False

        if self.x != agent.x:
            return False

        if self.y != agent.y:
            return False

        if self.bombs_left != agent.bombs_left:
            return False

        return True
