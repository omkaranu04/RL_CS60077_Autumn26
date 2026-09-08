import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grid import ACTIONS, Grid
from agent import Agent

ARROW_SYMBOLS = {"Up": " ^ ", "Down": " v ", "Left": " < ", "Right": " > "}

# -----------------------------
# User prompt helper functions
# -----------------------------
def prompt_int(prompt_text, default=None, min_val=None, max_val=None):
    while True:
        raw = input(prompt_text).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
            if min_val is not None and val < min_val:
                print(f"     Please enter an integer >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"     Please enter an integer <= {max_val}")
                continue
            return val
        except ValueError:
            print("     Please enter a valid integer")

def prompt_float(prompt_text, default=None, min_val=None, max_val=None):
    while True:
        raw = input(prompt_text).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"     Please enter a number >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"     Please enter a number <= {max_val}")
                continue
            return val
        except ValueError:
            print("     Please enter a valid number")

def prompt_coord(prompt_text, default=None, n=None, m=None):
    while True:
        raw = input(prompt_text).strip()
        if raw == "" and default is not None:
            return tuple(default)
        try:
            r, c = raw.split(",")
            r = int(r.strip())
            c = int(c.strip())
            if n is not None and m is not None:
                if not (0 <= r < n and 0 <= c < m):
                    print(f"     Please enter valid coordinates within the grid (0-{n-1}, 0-{m-1})")
                    continue
            return (r, c)
        except (ValueError, AttributeError):
            print("     Please enter coordinates as row,col eg. 1,1")

# -----------------------------
# Training Function
# -----------------------------
def train(env: Grid, agent: Agent, episodes, verbose_after=0):
    """
        Run Q-Learning for 'episode' episodes
        Return: (reward, step) history
    """
    rewards_per_episode, steps_per_episode = [], []
    for ep in range(1, episodes + 1):
        state = env.st
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < env.max_steps:
            action = agent.choose_action(state=state)
            next_state, reward, done = env.step(state=state, action_idx=action)
            agent.update(state=state, action=action, reward=reward, next_state=next_state, done=done)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_eps()                           # epsilon decay after 1 episode
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        # verbose printing option
        if verbose_after and ep % verbose_after == 0:
            print(f"  Episode {ep:5d}/{episodes} | "
                  f"reward={total_reward:8.1f} | steps={steps:4d} | "
                  f"epsilon={agent.eps:.3f}")

    return rewards_per_episode, steps_per_episode

def extract_policy(agent: Agent):
    """
        Return {state: best_action} -> for every non obstacle cell, based on the learned Q-Table
    """
    env = agent.env
    policy = {}
    for x in range(env.n):
        for y in range(env.m):
            if (x, y) in env.obs:
                continue
            policy[(x, y)] = int(np.argmax(agent.q[x, y]))
    return policy

def get_best_path(env: Grid, agent: Agent):
    """
        Greedily follow learned policy from start to goal
        Returns: (path, actions, step_rewards, total_reward, success)
        success=False -> if policy loops back to visited state without ever reaching goal
    """
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

        action = agent.choose_action(state=state, greedy=True)              # greedy=True -> pure exploitation
        next_state, reward, done = env.step(state=state, action_idx=action)
        actions.append(action)
        step_rewards.append(reward)
        total_reward += reward
        path.append(next_state)
        state = next_state

        if done:
            break

    success = (state == env.en)
    return path, actions, step_rewards, total_reward, success

def print_q_table(agent: Agent, rows, cols):
    """
        Print the final Q-Table: (states x actions) dimension
    """
    lines = []
    header = f"{'State':<10}" + "".join(f"{name.upper():>10}" for name in ACTIONS)
    lines.append(header)
    lines.append("-" * len(header))
    for x in range(rows):
        for y in range(cols):
            state_str = f"({x},{y})"
            q_vals = agent.q[x, y]
            row = f"{state_str:<10}" + "".join(f"{q:>10.2f}" for q in q_vals)
            lines.append(row)
    print("\n".join(lines))

def print_policy_grid(env: Grid, policy):
    """
        Print the learned policy as grid of actions (arrows)
    """
    for x in range(env.n):
        row = []
        for y in range(env.m):
            if (x, y) in env.obs:
                row.append(" X ")
            elif (x, y) == env.en:
                row.append(" G ")
            else:
                a = policy[(x, y)]
                row.append(ARROW_SYMBOLS[ACTIONS[a]])
        print("".join(row))

def format_path_table(path, actions, step_rewards, total_reward):
    """
        For the table formatting as shown in the Sample Output (in class)
    """
    col_widths = (5, 11, 9, 11)
    lines = []
    header = (f"{'Step':<{col_widths[0]}}{'From':<{col_widths[1]}}{'Action':<{col_widths[2]}}{'To':<{col_widths[3]}}{'Reward':>6}")
    lines.append(header)
    lines.append("-" * (sum(col_widths) + 6))
    for i, (action, reward) in enumerate(zip(actions, step_rewards), start=1):
        frm = str(path[i - 1])
        to = str(path[i])
        action_name = ACTIONS[action].upper()
        lines.append(f"{i:<{col_widths[0]}}{frm:<{col_widths[1]}}{action_name:<{col_widths[2]}}{to:<{col_widths[3]}}{reward:>6}")
    lines.append("-" * (sum(col_widths) + 6))
    lines.append(f"{'Total':<{sum(col_widths)}}{total_reward:>6}")
    return "\n".join(lines)

# -----------------------------
# Plot Functions
# -----------------------------
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
