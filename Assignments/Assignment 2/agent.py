import random
import numpy as np
from grid import ACTIONS

class Agent:
    """Q-Learning updated with standard TD(0) / Bellman update rule"""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, eps=0.1, eps_decay=1.0, eps_min=0.01, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.q = np.zeros((n_states, n_actions))    # all Q-values start at 0
        self.rng = random.Random(seed)              # rng, independent of global random

    def choose_action(self, state, greedy=False):
        """
            Epsilon-greedy action selection. greedy=True -> exploit (no exploration)
        """
        if (not greedy) and self.rng.random() < self.eps:
            return self.rng.randrange(self.n_actions)
        q_val = self.q[state]
        max_q = np.max(q_val)
        best_actions = np.flatnonzero(q_val == max_q)
        return int(self.rng.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        """
            Q-Learning table update
            Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a'[q(s', a')] - Q(s, a))
        """
        best_next = 0.0 if done else np.max(self.q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[state, action]
        self.q[state, action] += self.alpha * td_error

    def decay_eps(self):
        """
            Shrink the epsilon value, with the given floor 'eps_min'
        """
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def policy_grid(self, rows, cols):
        """
            Greedy actions, using the arrow symbols for the moves taken
        """
        arrows = {"Up": "^", "Down": "v", "Left": "<", "Right": ">"}
        grid = []
        for r in range(rows):
            row_syms = []
            for c in range(cols):
                s = r * cols + c
                a = int(np.argmax(self.q[s]))
                row_syms.append(arrows[ACTIONS[a]])
            grid.append(row_syms)
        return grid
