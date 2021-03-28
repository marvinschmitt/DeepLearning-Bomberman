from typing import List
import events as e


def reward_from_events(events: List[str]) -> float:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 500,
        # positive auxiliary rewards
        e.BOMB_DROPPED: 0,
        e.COIN_FOUND: 0,
        e.SURVIVED_ROUND: 0,
        e.CRATE_DESTROYED: 10,
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.INVALID_ACTION: 0.000,
        e.WAITED: 0.000,
        e.GOT_KILLED: -500,
        e.KILLED_SELF: -500
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
