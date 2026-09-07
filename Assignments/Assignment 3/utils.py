import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from board import Board, X, O, SYMBOLS, R_LOSE, R_DRAW
from agent import Agent

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

def play_episode(board: Board, agent_x: Agent, agent_o: Agent, learn=True, greedy=False, record_trace=False):
    """
        Plays one self-play episode start to finish.
        A single Board.step() only knows the mover's own outcome (win/valid/invalid),
        so once the game ends the opponent's own +100/-100/0 terminal reward is
        credited here, against the opponent's last (state, action) pair.
        Returns: (episode_reward, marks_placed, winner, draw, trace)
        episode_reward is a dict keyed by X/O; trace is a list of
        (player, action, cells_after) tuples when record_trace=True, else None.
    """
    state = board.reset()
    agents = {X: agent_x, O: agent_o}
    last = {X: None, O: None}
    episode_reward = {X: 0.0, O: 0.0}
    marks_placed = 0
    winner = None
    draw = False
    trace = [] if record_trace else None

    while True:
        if board.steps_taken >= board.max_steps:
            draw = True
            break

        player = board.current_player
        agent = agents[player]
        action = agent.choose_action(state, greedy=greedy)
        next_state, reward, done, info = board.step(action)

        if not info["valid_move"]:
            if learn:
                agent.update(state=state, action=action, reward=reward, next_state=next_state, done=False)
            episode_reward[player] += reward
            state = next_state
            continue

        marks_placed += 1
        episode_reward[player] += reward
        if learn:
            agent.update(state=state, action=action, reward=reward, next_state=next_state, done=done)
        last[player] = (state, action)
        state = next_state
        if record_trace:
            trace.append((player, action, tuple(board.cells)))

        if done:
            winner = info["winner"]
            draw = info["draw"]
            opponent = O if player == X else X
            if last[opponent] is not None:
                opp_reward = R_LOSE if winner is not None else R_DRAW
                if learn:
                    opp_state, opp_action = last[opponent]
                    agents[opponent].update(state=opp_state, action=opp_action, reward=opp_reward, next_state=state, done=True)
                episode_reward[opponent] += opp_reward
            break

    return episode_reward, marks_placed, winner, draw, trace

def train_self_play(board: Board, agent_x: Agent, agent_o: Agent, episodes, verbose_after=0):
    reward_x, reward_o, steps_per_episode, outcomes = [], [], [], []
    for ep in range(1, episodes + 1):
        ep_reward, marks_placed, winner, draw, _ = play_episode(board, agent_x, agent_o, learn=True)
        agent_x.decay_eps()
        agent_o.decay_eps()

        reward_x.append(ep_reward[X])
        reward_o.append(ep_reward[O])
        steps_per_episode.append(marks_placed)
        outcomes.append("X" if winner == X else "O" if winner == O else "draw")

        if verbose_after and ep % verbose_after == 0:
            recent = outcomes[-verbose_after:]
            x_rate = recent.count("X") / len(recent)
            o_rate = recent.count("O") / len(recent)
            draw_rate = recent.count("draw") / len(recent)
            print(f"  Episode {ep:6d}/{episodes} | "
                  f"reward_X={ep_reward[X]:7.1f} reward_O={ep_reward[O]:7.1f} | moves={marks_placed:2d} | "
                  f"last {len(recent)}: X={x_rate:.0%} O={o_rate:.0%} draw={draw_rate:.0%} | "
                  f"eps_X={agent_x.eps:.3f}")

    return reward_x, reward_o, steps_per_episode, outcomes

def print_q_sample(agent: Agent, board: Board, name, top_k=15):
    most_visited = sorted(agent.visit_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    print(f"{name} -- top {len(most_visited)} most-visited states (of {len(agent.visit_counts)} visited)")
    if not most_visited:
        print("  (no states visited yet)")
        return
    header = f"{'Visits':>7}  {'Board (row-major)':<{board.n_cells + 2}}" + "".join(f"{a:>7}" for a in range(board.n_actions))
    print(header)
    print("-" * len(header))
    for state, count in most_visited:
        cells = board.state_from_idx(state)
        board_flat = "".join(SYMBOLS[c] for c in cells)
        q_vals = agent.q[state]
        row = f"{count:>7}  {board_flat:<{board.n_cells + 2}}" + "".join(f"{v:7.1f}" for v in q_vals)
        print(row)

def format_trace(trace, board: Board):
    col_widths = (5, 8, 6)
    header = f"{'Move':<{col_widths[0]}}{'Player':<{col_widths[1]}}{'Cell':<{col_widths[2]}}Board After"
    lines = [header, "-" * (sum(col_widths) + 4 * board.size)]
    for i, (player, action, cells_after) in enumerate(trace, start=1):
        board_flat = " ".join(SYMBOLS[c] for c in cells_after)
        lines.append(f"{i:<{col_widths[0]}}{SYMBOLS[player]:<{col_widths[1]}}{action:<{col_widths[2]}}{board_flat}")
    return "\n".join(lines)

def plot_rewards(reward_x, reward_o, out_path):
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(reward_x) + 1), reward_x, linewidth=1, label="Agent X", color="royalblue")
    plt.plot(range(1, len(reward_o) + 1), reward_o, linewidth=1, label="Agent O", color="crimson")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode (both agents)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_steps(steps, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, linewidth=1, color="darkorange")
    plt.xlabel("Episode")
    plt.ylabel("Moves in Game")
    plt.title("Game Length (moves) per Episode")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_comparison(steps, outcomes, out_path):
    n = len(steps)
    window = max(1, n // 10)
    window = min(window, n)

    def rates(block):
        return (block.count("X") / len(block), block.count("O") / len(block), block.count("draw") / len(block))

    early_x_rate, early_o_rate, early_draw_rate = rates(outcomes[:window])
    final_x_rate, final_o_rate, final_draw_rate = rates(outcomes[-window:])
    early_steps_avg = float(np.mean(steps[:window]))
    final_steps_avg = float(np.mean(steps[-window:]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels = ["X wins", "O wins", "Draw"]
    x_pos = np.arange(len(labels))
    bar_w = 0.35
    axes[0].bar(x_pos - bar_w / 2, [early_x_rate, early_o_rate, early_draw_rate], bar_w, label="Early", color="salmon")
    axes[0].bar(x_pos + bar_w / 2, [final_x_rate, final_o_rate, final_draw_rate], bar_w, label="Final", color="seagreen")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels)
    axes[0].set_title(f"Outcome Rate\n(first {window} vs last {window} episodes)")
    axes[0].set_ylabel("Rate")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(["Early", "Final"], [early_steps_avg, final_steps_avg], color=["salmon", "seagreen"])
    axes[1].set_title(f"Avg Game Length\n(first {window} vs last {window} episodes)")
    axes[1].set_ylabel("Average Moves")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return {
        "window": window,
        "early_x_rate": early_x_rate, "final_x_rate": final_x_rate,
        "early_o_rate": early_o_rate, "final_o_rate": final_o_rate,
        "early_draw_rate": early_draw_rate, "final_draw_rate": final_draw_rate,
        "early_steps_avg": early_steps_avg, "final_steps_avg": final_steps_avg,
    }

def save_summary_json(path, summary):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
