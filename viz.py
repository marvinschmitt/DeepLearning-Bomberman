import matplotlib.pyplot as plt
import numpy as np
import pickle

import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_peaceful_double_axis():
    with open("results_checkpoints/results_35k/metrics.pt", 'rb') as f:
        r = pickle.load(f)
    scores = r["scores"]
    epsilons = r["epsilons"]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('game')
    ax1.set_ylabel('score', color=color)
    ax1.plot(np.convolve(scores, np.ones(1000)/1000, mode='valid'), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(np.convolve(scores, np.ones(10)/10, mode='valid'), color='gray', alpha=.20)
    ax1.set_ylim(0, 2.5)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('epsilon', color=color)  # we already handled the x-label with ax1
    ax2.plot(epsilons, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title("Running average reward: peaceful agent on 9x9 board")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("plots/peaceful_9x9_avg_score.png")
    plt.show()


def plot_results(filepath, title, savefname):
    with open(filepath, "rb") as f:
        metrics = pickle.load(f)
    losses, rewards = tuple(np.array(metrics).T)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    fig.suptitle(title)

    ax1.plot(losses)
    ax1.set_title("Loss")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")

    ax2.plot(rewards)
    ax2.plot(np.convolve(rewards, np.ones(100)/100), color="green")
    ax2.set_title("Average reward (past 100 ep)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average reward")

    plt.savefig(savefname)


if __name__ == '__main__':
    plot_results("results_checkpoints/checkpoints/checkpoint_50k/all_metrics.pickle",
                 "DQN Agent", "plots/dqn_results.png")
    plot_results("results_checkpoints/checkpoints/checkpoint_ppo/all_metrics.pickle",
                 "PPO Agent", "plots/ppo_results.png")
    plot_results("results_checkpoints/checkpoints/checkpoint_imitator/all_metrics.pickle",
                 "Imitator Task", "plots/imitator_results.png")
    plot_peaceful_double_axis()