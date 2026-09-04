import os
from qlearning import (Grid, Agent, train, extract_policy, 
                       get_best_path, print_q, format_path_table, 
                       plot_rewards, plot_steps, plot_comparison, 
                       plot_grid_path, ACTION_NAMES)

# A Default Grid as given in the Assignment PDF
DEFAULT_N = 6
DEFAULT_M = 5
DEFAULT_START = (0, 0)
DEFAULT_GOAL = (5, 4)
DEFAULT_OBS = [(0, 4), (1, 2), (2, 0), (2, 4), (3, 3), (4, 0), (5, 2)]
ARROW_SYMBOLS = {"Up": " ^ ", "Down": " v ", "Left": " < ", "Right": " > "}

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
            
def user_inputs():
    n = prompt_int("Enter number of rows (N): ", default=DEFAULT_N, min_val=1)
    m = prompt_int("Enter number of columns (M): ", default=DEFAULT_M, min_val=1)
    if n is None or m is None:
        return None, None, None, None, None, None, None, None, None
    start = prompt_coord("Enter Start position as row,col : ", default=DEFAULT_START, n=n, m=m)
    goal = prompt_coord("Enter Goal position as row,col : ", default=DEFAULT_GOAL, n=n, m=m)
    n_obs = prompt_int("Enter number of obstacles: ", default=len(DEFAULT_OBS), min_val=0, max_val=n*m-2)
    obs = []
    for i in range(n_obs):
        default = DEFAULT_OBS[i] if i < len(DEFAULT_OBS) else None
        obs.append(prompt_coord(f"Enter obstacle {i + 1} position as row,col : ", default=default, n=n, m=m))
    alpha = prompt_float("Enter learning rate (alpha): ", default=0.1, min_val=0.0, max_val=1.0)
    gamma = prompt_float("Enter discount factor (gamma): ", default=0.9, min_val=0.0, max_val=1.0)
    eps = prompt_float("Enter exploration rate (epsilon): ", default=0.2, min_val=0.0, max_val=1.0)
    episodes = prompt_int("Enter number of training episodes: ", default=1000, min_val=1)
    return n, m, start, goal, obs, alpha, gamma, eps, episodes

def print_policy_grid(env: Grid, policy):
    for x in range(env.n):
        row = []
        for y in range(env.m):
            if(x, y) in env.obs:
                row.append(" X ")
            elif (x, y) == env.en:
                row.append(" G ")
            else:
                a = policy[(x, y)]
                row.append(ARROW_SYMBOLS[ACTION_NAMES[a]])
        print("".join(row))
        
def main():
    (n, m, start, goal, obs, alpha, gamma, eps, episodes) = user_inputs()
    
    try:
        env = Grid(n=n, m=m, st=start, en=goal, obs=obs)
    except ValueError as e:
        print(f"\nInvalid environment configurations: {e}")
        return
    
    print("\nGrid Layout:")
    print(env.print_grid())
    
    agent = Agent(env=env, alpha=alpha, gamma=gamma, eps=eps)
    rewards, steps = train(env=env, agent=agent, episodes=episodes)
    
    print("\n=== Final Q-Table ===")
    print_q(agent=agent)
    
    path, actions, step_rewards, total_reward, success = get_best_path(env=env, agent=agent)
    print("\n=== Learned Optimal Path ===")
    if success:
        print(format_path_table(path=path, actions=actions, step_rewards=step_rewards, total_reward=total_reward))
    else:
        print("The policy did not reach the goal (loop/dead-end detected)")
        print("Partial path followed: " + " -> ".join(str(p) for p in path))
    print(f"\nNumber of steps: {len(actions)}")
    
    policy = extract_policy(agent=agent)
    print(f"\n=== Optimal Policy ===")
    print_policy_grid(env=env, policy=policy)
    
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    plot_grid_path(env=env, path=path, out_path=os.path.join(out_dir, "learned_optimal_path.png"))
    plot_rewards(rewards=rewards, out_path=os.path.join(out_dir, "reward_per_episode.png"))
    plot_steps(steps=steps, out_path=os.path.join(out_dir, "steps_per_episode.png"))
    stats = plot_comparison(rewards=rewards, steps=steps, out_path=os.path.join(out_dir, "early_vs_final_performance.png"))
    
    print("\n=== Comparison: Early vs Final Performance ===")
    print(f"(Averaged over the first {stats['window']} vs last {stats['window']} episodes)")
    print(f"  Average reward - early: {stats['early_reward_avg']:.2f}  final: {stats['final_reward_avg']:.2f}")
    print(f"  Average steps  - early: {stats['early_steps_avg']:.2f}   final: {stats['final_steps_avg']:.2f}")
    
if __name__ == "__main__":
    main()