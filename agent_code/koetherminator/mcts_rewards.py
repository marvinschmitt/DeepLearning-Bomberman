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
        e.CRATE_DESTROYED: 100,
        e.GOT_KILLED: -100
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            if event == e.GOT_KILLED:
                return game_rewards[e.GOT_KILLED]
            reward_sum += game_rewards[event]
    return reward_sum
