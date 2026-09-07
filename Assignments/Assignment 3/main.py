import argparse, os
from board import Board, X, O
from agent import Agent
import utils

DEMO_SIZE = 3
DEMO_ALPHA = 0.1
DEMO_GAMMA = 0.9
DEMO_EPS = 0.3

def build_from_args(args):
    if args.demo:
        print("Running in --demo mode with the classic 3x3 board")
        return DEMO_SIZE, DEMO_ALPHA, DEMO_GAMMA, DEMO_EPS, args.seed

    size = utils.prompt_int("Board size N (N x N grid, e.g. 3): ", default=DEMO_SIZE, min_val=3, max_val=5)
    alpha = utils.prompt_float("Learning rate alpha (e.g. 0.1): ", default=DEMO_ALPHA, min_val=0.0, max_val=1.0)
    gamma = utils.prompt_float("Discount factor gamma (e.g. 0.9): ", default=DEMO_GAMMA, min_val=0.0, max_val=1.0)
    eps = utils.prompt_float("Exploration rate epsilon (e.g. 0.3): ", default=DEMO_EPS, min_val=0.0, max_val=1.0)
    seed = utils.prompt_int("Random seed for training (blank for default): ", default=args.seed)

    return size, alpha, gamma, eps, seed

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--eps-decay", type=float, default=0.9995)
    args = parser.parse_args()

    size, alpha, gamma, eps, seed = build_from_args(args)
    episodes = args.episodes
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    board = Board(size=size)
    agent_x = Agent(board.n_states, board.n_actions, alpha=alpha, gamma=gamma, eps=eps, eps_decay=args.eps_decay, seed=seed)
    agent_o = Agent(board.n_states, board.n_actions, alpha=alpha, gamma=gamma, eps=eps, eps_decay=args.eps_decay, seed=seed + 1)

    print(f"\nBoard: {size}x{size}   Cells: {board.n_cells}   States: {board.n_states}")
    print(f"alpha={alpha} gamma={gamma} epsilon={eps} eps_decay={args.eps_decay} episodes={episodes}\n")
    print("Training (self-play)...")
    reward_x, reward_o, steps, outcomes = utils.train_self_play(
        board=board, agent_x=agent_x, agent_o=agent_o, episodes=episodes,
        verbose_after=max(1, episodes // 10))
    print("\nTraining complete\n")

    print("==== Q-table sample: Agent X ====")
    utils.print_q_sample(agent_x, board, name="Agent X")
    print("\n==== Q-table sample: Agent O ====")
    utils.print_q_sample(agent_o, board, name="Agent O")

    print("\n=== Learned Self-Play Trace (greedy, from empty board) ===")
    ep_reward, marks_placed, winner, draw, trace = utils.play_episode(
        board=board, agent_x=agent_x, agent_o=agent_o, learn=False, greedy=True, record_trace=True)
    print(utils.format_trace(trace, board))
    outcome_str = "Draw" if draw else ("X wins" if winner == X else "O wins")
    print(f"\nOutcome: {outcome_str}   Moves: {marks_placed}")
    print(f"Total reward - Agent X: {ep_reward[X]}   Agent O: {ep_reward[O]}")

    utils.plot_rewards(reward_x, reward_o, os.path.join(out_dir, "reward_per_episode.png"))
    utils.plot_steps(steps, os.path.join(out_dir, "steps_per_episode.png"))
    stats = utils.plot_comparison(steps, outcomes, os.path.join(out_dir, "early_vs_final_performance.png"))

    print("\n=== Comparison: Early vs Final Performance ===")
    print(f"(Over the first {stats['window']} vs last {stats['window']} episodes)")
    print(f"  X win rate       - early: {stats['early_x_rate']:.1%}  final: {stats['final_x_rate']:.1%}")
    print(f"  O win rate       - early: {stats['early_o_rate']:.1%}  final: {stats['final_o_rate']:.1%}")
    print(f"  Draw rate        - early: {stats['early_draw_rate']:.1%}  final: {stats['final_draw_rate']:.1%}")
    print(f"  Avg game length  - early: {stats['early_steps_avg']:.2f}  final: {stats['final_steps_avg']:.2f}")

    summary = {
        "board_size": size,
        "hyperparameters": {"alpha": alpha, "gamma": gamma, "epsilon": eps, "episodes": episodes, "epsilon_decay": args.eps_decay},
        "final_trace": {
            "moves": [{"player": "X" if p == X else "O", "cell": a} for p, a, _ in trace],
            "outcome": outcome_str, "moves_count": marks_placed,
            "reward_x": ep_reward[X], "reward_o": ep_reward[O],
        },
        "final_epsilon_x": agent_x.eps, "final_epsilon_o": agent_o.eps,
        "final_win_draw_rates": {
            "x_rate": stats["final_x_rate"], "o_rate": stats["final_o_rate"], "draw_rate": stats["final_draw_rate"],
        },
    }
    utils.save_summary_json(os.path.join(out_dir, "summary.json"), summary=summary)

if __name__ == "__main__":
    main()
