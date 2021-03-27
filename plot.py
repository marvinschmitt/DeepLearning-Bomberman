import pickle
import numpy as np
import matplotlib.pyplot as plt

# Create some mock data
with open("results_35k/metrics.pt", 'rb') as f:
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
plt.savefig("peaceful_9x9_avg_score.png")
plt.show()


