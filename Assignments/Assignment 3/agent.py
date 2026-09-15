import random
import numpy as np

class Agent:
    """Tabular Q-Learning Agent: Action Selection + Updates"""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, eps=0.1, eps_decay=1.0, eps_min=0.01, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min                      # floor for the epsilon value (after decay)
        self.q = np.zeros((n_states, n_actions))    # Q-Table -> begin with all zeros
        self.rng = random.Random(seed)              # dedicated rng per-agent
        self.visit_counts = {}                      # state -> number of moves made from it

    def choose_action(self, state, greedy=False):
        """
            Epsilon-greedy action selection, ties are broken randomly
        """
        if (not greedy) and self.rng.random() < self.eps:
            return self.rng.randrange(self.n_actions)       # explore: uniform random action
        q_val = self.q[state]
        max_q = np.max(q_val)
        # break ties randomly
        best_actions = np.flatnonzero(q_val == max_q)
        return int(self.rng.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        """
            Standard Q-Learning (off-policy TD) -> update using the max Q-value over next_state
        """
        best_next = 0.0 if done else np.max(self.q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[state, action]
        self.q[state, action] += self.alpha * td_error
        self.visit_counts[state] = self.visit_counts.get(state, 0) + 1

    def decay_eps(self):
        """epsilon decay respecting the floor"""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
