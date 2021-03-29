import numpy as np
from tensorflow.keras.models import load_model
from adapter.bomberman_adapter import BombermanEnvironment
from deep_bomber.agent_network import network
import pickle


class ReplayBuffer():
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)
    
    # state_ is next state
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1-int(done)
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, terminal


class Agent():
    def __init__(self, q_net, gamma, epsilon, batch_size,
                 input_dims, n_actions, epsilon_dec=1e-3, epsilon_end=0.01,
                 mem_size=1000000, fname='dqn_model_koetherminator.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_net = q_net
    
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_net(observation[np.newaxis,:])
            action = np.argmax(actions)
        
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            # only learn if buffer is full enough
            return
        
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        
        q_eval = self.q_net(states)
        q_next = self.q_net(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)*dones

        self.q_net.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
    
    def save_model(self):
        self.q_net.save(self.model_file)
        
    def load_model(self):
        self.q_net = load_model(self.model_file)


SAVE_EACH_GAMES = 100

if __name__ == '__main__':
    env = BombermanEnvironment(mode="no_bomb")
    lr = 0.001
    n_games = 50000
    q_net = network(lr=lr, n_actions=len(env.actions), 
                    input_dims=env.observation_shape)
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
