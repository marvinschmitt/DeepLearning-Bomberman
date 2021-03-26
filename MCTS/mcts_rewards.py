from typing import List
import events as e


def reward_from_events(events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        # positive auxiliary rewards
        e.BOMB_DROPPED: 0.001,
        # e.COIN_FOUND: 0.01,
        # e.SURVIVED_ROUND: 0.5,
        e.CRATE_DESTROYED: 0.1,
        e.MOVED_LEFT: 0.001,
        e.MOVED_RIGHT: 0.001,
        e.MOVED_UP: 0.001,
        e.MOVED_DOWN: 0.001,
        # negative auxiliary rewards
        e.INVALID_ACTION: -0.002,
        e.WAITED: -0.002,
        e.GOT_KILLED: -1,
        e.KILLED_SELF: -1
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
