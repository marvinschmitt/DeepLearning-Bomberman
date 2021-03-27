from adapter.bomberman_adapter import BombermanGame, BombermanEnvironment
from deep_bomber.dqn import Agent


def setup(self):
    self.agent = Agent(gamma=0.99, epsilon=1.0, lr=1e-3,
                       input_dims=(17, 17, 1), epsilon_dec=1e-6,
                       n_actions=6, mem_size=100000, batch_size=64,
                       epsilon_end=0.01, fname='dqn_model_koetherminator.h5')
    self.agent.load_model()
    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']


def act(self, game_state: dict):
    observation = BombermanGame.get_observation_from_state(game_state)
    action = self.agent.choose_action(observation)

    return self.actions[action]
