import csv, json, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from grid import ACTIONS, Grid
from agent import Agent

def gen_random_grid(rows, cols, st, en, seed=None):
    """
        Random integer reward generator for the grid for every cell for within limits
    """
    rng = random.Random(seed)
    grid = [[0] * cols for _ in range(rows)]
    choices = list(range(-10, 0)) + list(range(1, 11))
    for r in range(rows):
        for c in range(cols):
            if (r, c) in (tuple(st), tuple(en)):
                continue
            grid[r][c] = rng.choice(choices)
    return grid

def get_manual_grid(rows, cols, st, en):
    """
        Prompt for the input of the cell to the user
    """
    grid = [[0] * cols for _ in range(rows)]
    print("\nEnter the hidden reward value for each cell.")
    print("Allowed ranges: 1 to 10 (positive reward) or -1 to -10 (negative reward).")
    for r in range(rows):
        for c in range(cols):
            if (r, c) == tuple(st):
                print(f"    Cell ({r}, {c}) is Start -> skipped (value unused)")
                continue
            if (r, c) == tuple(en):
                print(f"    Cell ({r}, {c}) is Goal -> skipped (fixed +100 bonus)")
                continue

            while(True):
                raw = input(f"  Value for cell ({r}, {c}): ").strip()
                try:
                    val = int(raw)
                except ValueError:
                    print("     Please enter an integer")
                    continue
                if val == 0 or not (-10 <= val <= 10):
                    print("     Value must be in 1...10 or -10...-1 (not 0)")
                    continue
                grid[r][c] = val
                break
    return grid

# Prompt helper functions
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

# Main training loop
def train(env: Grid, agent: Agent, episodes, verbose_after=0):
    rewards_per_episode, steps_per_episode = [], []
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.choose_action(state=state)
            next_state, reward, done, info = env.step(action_idx=action)
            agent.update(state=state, action=action, reward=reward, next_state=next_state, done=done)
            state = next_state
            total_reward += reward

        agent.decay_eps() # decay the epsilon after an episode
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(env.steps_taken)

        if verbose_after and ep % verbose_after == 0:
            print(f"  Episode {ep:5d}/{episodes} | "
                  f"reward={total_reward:8.1f} | steps={env.steps_taken:4d} | "
                  f"epsilon={agent.eps:.3f}")

    return rewards_per_episode, steps_per_episode

def get_best_path(env: Grid, agent: Agent):
    """
        Extract the greedy learned policy from start to end (exploit since greedy)
    """
    state = env.reset()
    actions, step_rewards = [], []
    total_reward = 0.0
    success = False

    while True:
        available = env.available_actions()
        if not available:
            break
        q_values = agent.q[state]
        best_action = max(available, key=lambda a: q_values[a])
        next_state, reward, done, info = env.step(best_action)
        actions.append(best_action)
        step_rewards.append(reward)
        total_reward += reward
        state = next_state

        if info["reached_goal"]:
            success = True
        if done:
            break

    return list(env.path), actions, step_rewards, total_reward, success

# output helper functions
def save_q_table_csv(agent: Agent, rows, cols, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "state"] + ACTIONS)
        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                writer.writerow([r, c, s] + list(agent.q[s]))

def print_q_table(agent: Agent, rows, cols):
    header = "State (r,c)   " + "   ".join(f"{a:>7s}" for a in ACTIONS)
    print(header)
    print("-" * len(header))
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            vals = "   ".join(f"{v:7.2f}" for v in agent.q[s])
            print(f"({r:2d},{c:2d})       {vals}")

def print_policy_grid(agent: Agent, rows, cols, st, en):
    grid = agent.policy_grid(rows, cols)
    print()
    for r in range(rows):
        row_str = []
        for c in range(cols):
            if (r, c) == tuple(st):
                row_str.append(" S ")
            elif (r, c) == tuple(en):
                row_str.append(" G ")
            else:
                row_str.append(f" {grid[r][c]} ")
        print("".join(row_str))
    print()

def format_path_table(path, actions, step_rewards, total_reward):
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

# plotting helper functions
def plot_rewards(rewards, out_path):
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(rewards) + 1), rewards, linewidth=1, label="reward per episode")
    if len(rewards) >= 10:
        window = max(1, len(rewards) // 20)
        avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window, len(avg) + window), avg, linewidth=2, label=f"rolling avg (w={window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend()
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
    rows, cols = env.n, env.m
    fig, ax = plt.subplots(figsize=(cols * 1.1 + 1, rows * 1.1 + 1))
    im = ax.imshow(env.grid_vals.astype(float), cmap="RdYlGn", vmin=-10, vmax=10)
    for r in range(rows):
        for c in range(cols):
            if (r, c) == env.st:
                text = "S"
            elif (r, c) == env.en:
                text = "G"
            else:
                text = str(int(env.grid_vals[r, c]))
            ax.text(c, r, text, ha="center", va="center",
                    fontsize=11, fontweight="bold")

    if path and len(path) > 1:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="blue", linewidth=2.5, marker="o",
                markersize=6, alpha=0.8)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title("Learned Optimal Path (S=start, G=goal)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="hidden cell value")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_summary_json(path, summary):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
