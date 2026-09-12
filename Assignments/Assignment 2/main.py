import argparse, os
from grid import Grid
from agent import Agent
import utils

# From the assignment (works as default input) as well as when --demo is used
DEMO_GRID = [
    [0, 9, -6, -2, 1],
    [-9, 2, -5, 1, 1],
    [1, 10, -5, 2, -2],
    [-3, 5, 2, 5, -1],
    [-5, -2, 4, 10, 2],
    [-2, 1, 2, 3, 0],
]
DEMO_START = (0, 0)
DEMO_GOAL = (5, 4)

def build_from_args(args):
    if args.demo:
        print("Running in --demo mode with example grid from the PDF")
        return (DEMO_GRID, DEMO_START, DEMO_GOAL, 0.1, 0.9, 0.2, 500, args.seed)

    # taking all the inputs (or retorting to defaults)
    rows = utils.prompt_int("Number of rows (N): ", default=len(DEMO_GRID), min_val=1)
    cols = utils.prompt_int("Number of columns (M): ", default=len(DEMO_GRID[0]), min_val=1)
    st = utils.prompt_coord("Start position (row,col): ", default=DEMO_START, n=rows, m=cols)
    en = utils.prompt_coord("Goal position (row,col): ", default=DEMO_GOAL, n=rows, m=cols)
    while st == en:
        print("     Goal must differ from Start.")
        en = utils.prompt_coord("Goal position (row,col): ", default=DEMO_GOAL, n=rows, m=cols)

    mode = input("Enter grid values manually or randomly? [m/r] (default r): ").strip().lower()
    if mode == "m":
        grid = utils.get_manual_grid(rows, cols, st, en)
    else:
        grid_seed = utils.prompt_int("Random seed for grid (blank for default): ", default=42)
        grid = utils.gen_random_grid(rows, cols, st, en, seed=grid_seed)
        print("\nGenerated hidden grid: ")
        for row in grid:
            print(row)

    alpha = utils.prompt_float("Learning rate alpha (e.g. 0.1): ", default=0.1, min_val=0.0, max_val=1.0)
    gamma = utils.prompt_float("Discount factor gamma (e.g. 0.9): ", default=0.9, min_val=0.0, max_val=1.0)
    eps = utils.prompt_float("Exploration rate epsilon (e.g. 0.2): ", default=0.2, min_val=0.0, max_val=1.0)
    episodes = utils.prompt_int("Number of training episodes (e.g. 500): ", default=500, min_val=1)
    seed = utils.prompt_int("Random seed for training (blank for default): ", default=args.seed)

    return grid, st, en, alpha, gamma, eps, episodes, seed

def main():
    # command line flags
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--eps-decay", type=float, default=1.0)
    args = parser.parse_args()

    grid, st, en, alpha, gamma, eps, episodes, seed = build_from_args(args)
    if args.episodes:
        episodes = args.episodes
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # build the environment and agent and train
    env = Grid(grid_vals=grid, st=st, en=en)
    agent = Agent(env.n_states, env.n_actions, alpha=alpha, gamma=gamma, eps=eps, eps_decay=args.eps_decay, seed=seed)
    print(f"\nGrid: {env.n}x{env.m}   Start={st}   Goal={en}")
    print(f"alpha={alpha} gamma={gamma} epsilon={eps} episodes={episodes} max_steps/episode={env.max_steps}\n")
    print("Training...")
    rewards, steps = utils.train(env=env, agent=agent, episodes=episodes, verbose_after=max(1, episodes // 10))
    print("\nTraining complete\n")

    print("==== Final Q-table ====")
    utils.print_q_table(agent=agent, rows=env.n, cols=env.m)

    # get the actual best path
    path, actions, step_rewards, total_reward, success = utils.get_best_path(env=env, agent=agent)
    print("\n=== Learned Optimal Path ===")
    if success:
        print(utils.format_path_table(path=path, actions=actions, step_rewards=step_rewards, total_reward=total_reward))
    else:
        print("The policy did not reach the goal (loop/dead-end detected)")
        print("Partial path followed: " + " -> ".join(str(p) for p in path))
    print(f"\nNumber of steps: {len(actions)}")

    print(f"\n=== Optimal Policy ===")
    utils.print_policy_grid(agent, env.n, env.m, st, en)

    # save all the required things
    utils.save_q_table_csv(agent, env.n, env.m, os.path.join(out_dir, "q_table.csv"))
    utils.plot_rewards(rewards, os.path.join(out_dir, "reward_per_episode.png"))
    utils.plot_steps(steps, os.path.join(out_dir, "steps_per_episode.png"))
    utils.plot_grid_path(env=env, path=path, out_path=os.path.join(out_dir, "learned_optimal_path.png"))
    stats = utils.plot_comparison(rewards, steps, os.path.join(out_dir, "early_vs_final_performance.png"))

    print("\n=== Comparison: Early vs Final Performance ===")
    print(f"(Averaged over the first {stats['window']} vs last {stats['window']} episodes)")
    print(f"  Average reward - early: {stats['early_reward_avg']:.2f}  final: {stats['final_reward_avg']:.2f}")
    print(f"  Average steps  - early: {stats['early_steps_avg']:.2f}   final: {stats['final_steps_avg']:.2f}")

    summary = {
        "rows": env.n, "cols": env.m, "start": st, "goal": en,
        "hyperparameters": {
            "alpha": alpha, "gamma": gamma, "epsilon": eps, "episodes": episodes, "epsilon_decay": args.eps_decay
        },
        "best_path": path, "total_reward": total_reward, "steps": len(actions), "reached_goal": success, "final_epsilon": agent.eps,
        "reward_last_episode": rewards[-1], "reward_best_episode": max(rewards)
    }
    utils.save_summary_json(os.path.join(out_dir, "summary.json"), summary=summary)

if __name__ == "__main__":
    main()
