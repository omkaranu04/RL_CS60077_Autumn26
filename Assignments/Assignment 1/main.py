import argparse, os
from grid import Grid
from agent import Agent
import utils

# Example grid from the given PDF, used with --demo and the fallback default when interative prompt is left blank
DEMO_N = 6
DEMO_M = 5
DEMO_START = (0, 0)
DEMO_GOAL = (5, 4)
DEMO_OBS = [(0, 4), (1, 2), (2, 0), (2, 4), (3, 3), (4, 0), (5, 2)]

def build_from_args(args):
    """
        Return (n, m, start, goal, obstacles, alpha, gamma, epsilon, seed) -> depending if demo/interactive
    """
    if args.demo:
        print("Running in --demo mode with example grid from the PDF")
        return DEMO_N, DEMO_M, DEMO_START, DEMO_GOAL, DEMO_OBS, 0.1, 0.9, 0.2, args.seed

    n = utils.prompt_int("Enter number of rows (N): ", default=DEMO_N, min_val=1)
    m = utils.prompt_int("Enter number of columns (M): ", default=DEMO_M, min_val=1)
    st = utils.prompt_coord("Enter Start position (row,col): ", default=DEMO_START, n=n, m=m)
    en = utils.prompt_coord("Enter Goal position (row,col): ", default=DEMO_GOAL, n=n, m=m)
    while st == en:     # re-prompt until start and goal are not different
        print("     Goal must differ from Start.")
        en = utils.prompt_coord("Enter Goal position (row,col): ", default=DEMO_GOAL, n=n, m=m)
    n_obs = utils.prompt_int("Enter number of obstacles: ", default=len(DEMO_OBS), min_val=0, max_val=n * m - 2)
    obs = []
    for i in range(n_obs):
        default = DEMO_OBS[i] if i < len(DEMO_OBS) else None
        obs.append(utils.prompt_coord(f"Enter obstacle {i + 1} position (row,col): ", default=default, n=n, m=m))
    alpha = utils.prompt_float("Learning rate alpha (e.g. 0.1): ", default=0.1, min_val=0.0, max_val=1.0)
    gamma = utils.prompt_float("Discount factor gamma (e.g. 0.9): ", default=0.9, min_val=0.0, max_val=1.0)
    eps = utils.prompt_float("Exploration rate epsilon (e.g. 0.2): ", default=0.2, min_val=0.0, max_val=1.0)
    seed = utils.prompt_int("Random seed for training (blank for default): ", default=args.seed)

    return n, m, st, en, obs, alpha, gamma, eps, seed

def main():
    # --episodes, --seed, --eps-decay, --output-dir are optional CLI flags (you can enter in interactive also)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--eps-decay", type=float, default=1.0)
    args = parser.parse_args()

    # 1. Collect configuration and build the environment
    n, m, st, en, obs, alpha, gamma, eps, seed = build_from_args(args)
    episodes = args.episodes
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    try:
        env = Grid(n=n, m=m, st=st, en=en, obs=obs)
    except ValueError as e:
        print(f"\nInvalid environment configuration: {e}")
        return

    print("\nGrid Layout:")
    print(env.print_grid())

    # 2. Train the Q-Learning Agent
    agent = Agent(env=env, alpha=alpha, gamma=gamma, eps=eps, eps_decay=args.eps_decay, seed=seed)
    print(f"\nGrid: {env.n}x{env.m}   Start={st}   Goal={en}")
    print(f"alpha={alpha} gamma={gamma} epsilon={eps} episodes={episodes} max_steps/episode={env.max_steps}\n")
    print("Training...")
    rewards, steps = utils.train(env=env, agent=agent, episodes=episodes, verbose_after=max(1, episodes // 10))
    print("\nTraining complete\n")

    # 3. Display 1: final Q-table
    print("==== Final Q-table ====")
    utils.print_q_table(agent=agent, rows=env.n, cols=env.m)

    # 4. Find 2/3/4 + Display 2: best path, total reward, number of steps
    path, actions, step_rewards, total_reward, success = utils.get_best_path(env=env, agent=agent)
    print("\n=== Learned Optimal Path ===")
    if success:
        print(utils.format_path_table(path=path, actions=actions, step_rewards=step_rewards, total_reward=total_reward))
    else:
        print("The policy did not reach the goal (loop/dead-end detected)")
        print("Partial path followed: " + " -> ".join(str(p) for p in path))
    print(f"\nNumber of steps: {len(actions)}")

    # 5. Find 1: Optimal Policy learned by Q-Learning
    policy = utils.extract_policy(agent=agent)
    print(f"\n=== Optimal Policy ===")
    utils.print_policy_grid(env=env, policy=policy)

    # 6. Display 3/4/5: save all required plots to 'out_dir'
    utils.plot_grid_path(env=env, path=path, out_path=os.path.join(out_dir, "learned_optimal_path.png"))
    utils.plot_rewards(rewards=rewards, out_path=os.path.join(out_dir, "reward_per_episode.png"))
    utils.plot_steps(steps=steps, out_path=os.path.join(out_dir, "steps_per_episode.png"))
    stats = utils.plot_comparison(rewards=rewards, steps=steps, out_path=os.path.join(out_dir, "early_vs_final_performance.png"))

    print("\n=== Comparison: Early vs Final Performance ===")
    print(f"(Averaged over the first {stats['window']} vs last {stats['window']} episodes)")
    print(f"  Average reward - early: {stats['early_reward_avg']:.2f}  final: {stats['final_reward_avg']:.2f}")
    print(f"  Average steps  - early: {stats['early_steps_avg']:.2f}   final: {stats['final_steps_avg']:.2f}")

if __name__ == "__main__":
    main()
