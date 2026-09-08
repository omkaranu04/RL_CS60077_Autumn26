import random
import numpy as np
from grid import ACTIONS, Grid

class Agent:
    """Tabular Q-Learning agent with epsilon-greedy exploration"""
    def __init__(self, env: Grid, alpha, gamma, eps, eps_decay=1.0, eps_min=0.01, seed=None):
        self.env = env
        self.alpha = alpha                                  # learning rate
        self.gamma = gamma                                  # discount factor
        self.eps = eps                                      # current exploration rate
        self.eps_decay = eps_decay                          # multiplicative exploration decay applied per episode
        self.eps_min = eps_min                              # floor for the exploration rate
        self.q = np.zeros((env.n, env.m, len(ACTIONS)))     # Q-table: [row, col, action]
        self.rng = random.Random(seed)                      # Eaxh agent has own rng, so training is reproducible given seed

    def choose_action(self, state, greedy=False):
        """
            Epsilon-greedy action selection
                if greedy=True -> always exploit (affects exploration)
        """
        if (not greedy) and self.rng.random() < self.eps:
            return self.rng.randrange(len(ACTIONS))

        x, y = state
        q_val = self.q[x, y]
        max_q = np.max(q_val)
        best_actions = np.flatnonzero(q_val == max_q)       # break ties among equally-good actions randomly
        return int(self.rng.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        """
            Q-Learning Update:
                Q(s, a) = Q(s, a) + alpha * (r + gamma * ([max_a'] Q(s', a')) - Q(s, a))
        """
        x, y = state
        nx, ny = next_state
        best_next_q = 0.0 if done else np.max(self.q[nx, ny])   # no bootstrap from terminal state
        td_target = reward + self.gamma * best_next_q           # TD target
        td_error = td_target - self.q[x, y, action]             # TD error
        self.q[x, y, action] += self.alpha * td_error

    def decay_eps(self):
        """
            Shrink epsilon by eps_decay, floor at eps_min, after every episode
        """
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
