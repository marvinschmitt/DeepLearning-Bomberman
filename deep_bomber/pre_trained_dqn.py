import numpy as np
from adapter.bomberman_adapter import BombermanEnvironment
from deep_bomber.dqn import Agent
from deep_bomber.agent_network import network
import pickle

SAVE_EACH_GAMES = 100

if __name__ == '__main__':
    env = BombermanEnvironment()
    lr = 0.01
    n_games = 50000
    q_net = network(lr=lr, n_actions=len(env.actions), 
                    input_dims=env.observation_shape)
    q_net.load_weights('pre_training/best-network.hdf5')
    agent = Agent(q_net=q_net, input_dims=env.observation_shape, n_actions=len(env.actions),
                  gamma=0.99, epsilon=1.0, epsilon_dec=1e-6, 
                  mem_size=100000, batch_size=64, epsilon_end=0.01)
    scores = []
    eps_history = []

    for i in range(1, n_games + 1):
        done = False
        score = 0
        observation, reward = env.reset()
        turn = 0

        while not done:
            action = agent.choose_action(observation)

            observation_, reward = env.step(action)
            done = env.is_finished()
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            
            observation = observation_
            agent.learn()
            turn += 1
            
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f'game: {i}, score: {score:.4f}, avg_score: {avg_score:.4f}, epsilon: {agent.epsilon:.4f}, num_turns: {turn}')

        if i % SAVE_EACH_GAMES == 0:
            agent.save_model()
            results = {'scores': scores, 'epsilons': eps_history}
            with open("metrics.pt", "wb") as file:
                pickle.dump(results, file)