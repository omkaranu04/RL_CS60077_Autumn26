import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

R_GOAL = 100
R_VALID = -1
R_INVALID = -10
R_OBS = -100

class Grid:
    def __init__(self, n, m, st, en, obs):
        self.n = n
        self.m = m
        self.st = tuple(st)
        self.en = tuple(en)
        self.obs = set(tuple(o) for o in obs)
        self.actions = list(range(len(ACTION_NAMES)))
        self.max_steps = 2 * n * m
        self._validate()
        
    def _validate(self):
        if not self.check(self.st):
            raise ValueError(f"Start position {self.st} is outside the grid")
        if not self.check(self.en):
            raise ValueError(f"Goal position {self.en} is outside the grid")
        if self.st in self.obs or self.en in self.obs:
            raise ValueError("Start/Goal position cannot be an obstacle")
        for o in self.obs:
            if not self.check(o):
                raise ValueError(f"Obstacle {o} is outside the grid")
            
    def check(self, pos):
        r, c = pos
        return 0 <= r < self.n and 0 <= c < self.m
    
    def step(self, state, action):
        """
            Apply 'action' from 'state'
            Returns: (next_state, reward, done)
            Rules:
                - move outside grid     -> stay in place, reward = -10
                - move into obstacle    -> stay in place, reward = -100
                - move into goal        -> episode ends,  reward = +100
                - any other move        -> valid move,    reward = -1
        """
        x, y = state
        dx, dy = ACTIONS[action]
        next_state = (x + dx, y + dy)
        if not self.check(next_state):
            return state, R_INVALID, False
        if next_state in self.obs:
            return state, R_OBS, False
        if next_state == self.en:
            return next_state, R_GOAL, True
        return next_state, R_VALID, False
    
    def print_grid(self):
        lines = []
        for x in range(self.n):
            row = []
            for y in range(self.m):
                pos = (x, y)
                if pos == self.st:
                    row.append("S")
                elif pos == self.en:
                    row.append("G")
                elif pos in self.obs:
                    row.append("X")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)
    
class Agent:
    def __init__(self, env: Grid, alpha, gamma, eps):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.q = np.zeros((env.n, env.m, len(env.actions)))
        
    def action(self, state, greedy=False):
        """
            If greedy=True -> exploit
        """
        if(not greedy) and random.random() < self.eps:
            return random.choice(self.env.actions)
        
        x, y, = state
        q_val = self.q[x, y]
        max_q = np.max(q_val)
        best_actions = np.flatnonzero(q_val == max_q)
        return int(random.choice(best_actions))
    
    def update(self, state, action, reward, next_state, done):
        x, y = state
        nx, ny = next_state
        best_next_q = 0.0 if done else np.max(self.q[nx, ny])
        t1 = reward + self.gamma * best_next_q
        t2 = t1 - self.q[x, y, action]
        self.q[x, y, action] += self.alpha * t2
        
def train(env: Grid, agent: Agent, episodes: int):
    rewards_per_episode, steps_per_episode = [], []
    for _ in range(episodes):
        state = env.st
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < env.max_steps:
            action = agent.action(state=state)
            next_state, reward, done = env.step(state=state, action=action)
            agent.update(state=state, action=action, reward=reward, next_state=next_state, done=done)
            state = next_state
            total_reward += reward
            steps += 1
            
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
    
    return rewards_per_episode, steps_per_episode

def extract_policy(agent: Agent):
    env = agent.env
    policy = {}
    for x in range(env.n):
        for y in range(env.m):
            if(x, y) in env.obs:
                continue
            policy[(x, y)] = int(np.argmax(agent.q[x, y]))
    return policy

def get_best_path(env: Grid, agent: Agent):
    max_len = env.max_steps
    state = env.st
    path = [state]
    actions = []
    step_rewards = []
    total_reward = 0
    visited = set()
    for _ in range(max_len):
        if state == env.en:
            break
        if state in visited:
            return path, actions, step_rewards, total_reward, False
        visited.add(state)
        
        action = agent.action(state=state, greedy=True)
        next_state, reward, done = env.step(state=state, action=action)
        actions.append(action)
        step_rewards.append(reward)
        total_reward += reward
        path.append(next_state)
        state = next_state
        
        if done:
            break
        
    success = (state == env.en)
    return path, actions, step_rewards, total_reward, success

def print_q(agent: Agent):
    env = agent.env
    lines = []
    header = f"{'State':<10}" + "".join(f"{name.upper():>10}" for name in ACTION_NAMES)
    lines.append(header)
    lines.append("-" * len(header))
    for x in range(env.n):
        for y in range(env.m):
            state_str = f"({x},{y})"
            q_vals = agent.q[x, y]
            row = f"{state_str:<10}" + "".join(f"{q:>10.2f}" for q in q_vals)
            lines.append(row)
    print("\n".join(lines))
    
def format_path_table(path, actions, step_rewards, total_reward):
    col_widths = (5, 11, 9, 11)
    lines = []
    header = (f"{'Step':<{col_widths[0]}}{'From':<{col_widths[1]}}{'Action':<{col_widths[2]}}{'To':<{col_widths[3]}}{'Reward':>6}")
    lines.append(header)
    lines.append("-" * (sum(col_widths) + 6))
    for i, (action, reward) in enumerate(zip(actions, step_rewards), start=1):
        frm = str(path[i - 1])
        to = str(path[i])
        action_name = ACTION_NAMES[action].upper()
        lines.append(f"{i:<{col_widths[0]}}{frm:<{col_widths[1]}}{action_name:<{col_widths[2]}}{to:<{col_widths[3]}}{reward:>6}")
    lines.append("-" * (sum(col_widths) + 6))
    lines.append(f"{'Total':<{sum(col_widths)}}{total_reward:>6}")
    return "\n".join(lines)
    
def plot_rewards(rewards, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(rewards, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_steps(steps, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, linewidth=1, color="darkorange")
    plt.xlabel("Episode")
    plt.ylabel("Steps Taken")
    plt.title("Steps per Episode")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
def plot_comparison(rewards, steps, out_path):
    n = len(rewards)
    window = max(1, n // 10)
    window = min(window, n)
    
    early_reward_avg = float(np.mean(rewards[:window]))
    final_reward_avg = float(np.mean(rewards[-window:]))
    early_steps_avg = float(np.mean(steps[:window]))
    final_steps_avg = float(np.mean(steps[-window:]))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(["Early", "Final"], [early_reward_avg, final_reward_avg], color=["salmon", "seagreen"])
    axes[0].set_title(f"Avg Reward\n(first {window} vs last {window} episodes)")
    axes[0].set_ylabel("Average Total Reward")
    axes[0].grid(True, axis="y", alpha=0.3)
 
    axes[1].bar(["Early", "Final"], [early_steps_avg, final_steps_avg], color=["salmon", "seagreen"])
    axes[1].set_title(f"Avg Steps\n(first {window} vs last {window} episodes)")
    axes[1].set_ylabel("Average Steps")
    axes[1].grid(True, axis="y", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
 
    return {
        "window": window,
        "early_reward_avg": early_reward_avg,
        "final_reward_avg": final_reward_avg,
        "early_steps_avg": early_steps_avg,
        "final_steps_avg": final_steps_avg,
    }
    
def plot_grid_path(env: Grid, path, out_path):
    fig, ax = plt.subplots(figsize=(max(4, env.m), max(4, env.n)))

    for r in range(env.n):
        for c in range(env.m):
            pos = (r, c)
            if pos in env.obs:
                color = "black"
            elif pos == env.st:
                color = "mediumseagreen"
            elif pos == env.en:
                color = "crimson"
            else:
                color = "white"
            y = env.n - 1 - r
            rect = plt.Rectangle((c, y), 1, 1, facecolor=color, edgecolor="gray")
            ax.add_patch(rect)

    if path:
        xs = [p[1] + 0.5 for p in path]
        ys = [env.n - 1 - p[0] + 0.5 for p in path]
        ax.plot(xs, ys, marker="o", color="royalblue", linewidth=2, markersize=6, zorder=5)
        
    ax.set_xlim(0, env.m)
    ax.set_ylim(0, env.n)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.m + 1))
    ax.set_yticks(range(env.n + 1))
    ax.set_title("Learned Optimal Path (S=green, G=red, X=obstacle)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()