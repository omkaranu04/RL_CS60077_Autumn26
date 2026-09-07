import random
import numpy as np
from grid import ACTIONS, Grid

class Agent:
    def __init__(self, env: Grid, alpha, gamma, eps, eps_decay=1.0, eps_min=0.01, seed=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.q = np.zeros((env.n, env.m, len(ACTIONS)))
        self.rng = random.Random(seed)

    def choose_action(self, state, greedy=False):
        if (not greedy) and self.rng.random() < self.eps:
            return self.rng.randrange(len(ACTIONS))

        x, y = state
        q_val = self.q[x, y]
        max_q = np.max(q_val)
        best_actions = np.flatnonzero(q_val == max_q)
        return int(self.rng.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        x, y = state
        nx, ny = next_state
        best_next_q = 0.0 if done else np.max(self.q[nx, ny])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q[x, y, action]
        self.q[x, y, action] += self.alpha * td_error

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
