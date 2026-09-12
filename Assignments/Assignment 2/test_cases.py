import sys
from grid import Grid, ACTIONS
from agent import Agent
import utils

"""
Defined manual test cases and tested them against the Agent
"""

def trace_actions(env, action_names):
    env.reset()
    total_reward = 0.0
    reached_goal = False
    for name in action_names:
        _, reward, done, info = env.step(ACTIONS.index(name))
        total_reward += reward
        if info["reached_goal"]:
            reached_goal = True
    return list(env.path), total_reward, env.steps_taken, reached_goal

# DFS will give the brute force optimal solution
def brute_force_optimal(env):
    best = {"reward": float("-inf"), "path": None}
    deltas = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
    def dfs(pos, visited, reward, path):
        if pos == env.en:
            if reward > best["reward"]:
                best["reward"] = reward
                best["path"] = list(path)
            return
        for dr, dc in deltas.values():
            new_pos = (pos[0] + dr, pos[1] + dc)
            if env.check(new_pos) and new_pos not in visited:
                step_reward = -5 + (100 if new_pos == env.en else int(env.grid_vals[new_pos]))
                visited.add(new_pos)
                path.append(new_pos)
                dfs(new_pos, visited, reward + step_reward, path)
                path.pop()
                visited.remove(new_pos)

    dfs(env.st, {env.st}, 0, [env.st])
    return best["reward"], best["path"]

def run_qlearning(env, episodes, epsilon, epsilon_decay, seed):
    agent = Agent(env.n_states, env.n_actions, alpha=0.1, gamma=0.9, eps=epsilon, eps_decay=epsilon_decay, seed=seed)
    utils.train(env, agent, episodes)
    path, actions, step_rewards, total_reward, success = utils.get_best_path(env, agent)
    return path, total_reward, len(actions), success


PASS, FAIL = "PASS", "FAIL"
results = []

def record(name, ok, lines):
    results.append((name, PASS if ok else FAIL))
    print(f"[{PASS if ok else FAIL}] {name}")
    for line in lines:
        print(f"       {line}")
    print()
    
def scenario_pdf_example():
    name = "Scenario 1: PDF example grid (6x5)"
    grid = [
        [0, 9, -6, -2, 1],
        [-9, 2, -5, 1, 1],
        [1, 10, -5, 2, -2],
        [-3, 5, 2, 5, -1],
        [-5, -2, 4, 10, 2],
        [-2, 1, 2, 3, 0],
    ]
    start, goal = (0, 0), (5, 4)
    env = Grid(grid, start, goal)
    hand_actions = ["Right", "Down", "Down", "Down", "Right", "Right", "Down", "Down", "Right"]
    hand_expected_reward, hand_expected_steps = 101, 9
 
    path, reward, steps, reached_goal = trace_actions(env, hand_actions)
    ok = (reached_goal and reward == hand_expected_reward
          and steps == hand_expected_steps)
    lines = [
        f"hand-traced path reward = {reward} (expected {hand_expected_reward})",
        f"hand-traced steps       = {steps} (expected {hand_expected_steps})",
    ]
 
    bf_reward, _ = brute_force_optimal(env)
    matches = bf_reward == hand_expected_reward
    lines.append(f"brute-force optimal reward = {bf_reward} "
                  f"({'matches hand calc' if matches else 'DOES NOT MATCH hand calc'})")
    ok = ok and matches
 
    q_path, q_reward, q_steps, q_goal = run_qlearning(
        env, episodes=800, epsilon=0.3, epsilon_decay=0.995, seed=42)
    lines.append(f"Q-learning greedy reward = {q_reward} "
                  f"({q_reward / bf_reward:.0%} of optimal), steps={q_steps}, "
                  f"goal reached={q_goal}")
    ok = ok and q_goal and (q_reward >= 0.9 * bf_reward)
 
    record(name, ok, lines)

def scenario_trivial_2x2():
    name = "Scenario 2: trivial 2x2 grid (both possible routes hand-checked)"
    grid = [[0, 5], [3, 0]]
    start, goal = (0, 0), (1, 1)
    env = Grid(grid, start, goal)
 
    _, reward_a, _, goal_a = trace_actions(env, ["Right", "Down"])
    _, reward_b, _, goal_b = trace_actions(env, ["Down", "Right"])
    lines = [
        f"Route A (Right,Down): reward={reward_a} (expected 95)",
        f"Route B (Down,Right): reward={reward_b} (expected 93)",
    ]
    ok = (goal_a and goal_b and reward_a == 95 and reward_b == 93
          and reward_a > reward_b)
 
    q_path, q_reward, q_steps, q_goal = run_qlearning(
        env, episodes=300, epsilon=0.3, epsilon_decay=0.99, seed=1)
    lines.append(f"Q-learning greedy path={q_path}, reward={q_reward}, "
                  f"goal reached={q_goal}")
    # State space is tiny (4 cells) -- expect the exact optimum, not just "close".
    ok = ok and q_goal and (q_reward == 95)
 
    record(name, ok, lines)
    
def scenario_forced_corridor():
    name = "Scenario 3: 1x5 forced corridor (tests invalid-move handling)"
    grid = [[0, -1, 5, -2, 0]]
    start, goal = (0, 0), (0, 4)
    env = Grid(grid, start, goal)
 
    path, reward, steps, reached_goal = trace_actions(
        env, ["Right", "Right", "Right", "Right"])
    lines = [f"only-possible-path reward = {reward} (expected 82), steps={steps}"]
    ok = reached_goal and reward == 82 and steps == 4
 
    env.reset()
    invalid_checks = []
    for action_name in ["Up", "Down", "Left"]:
        _, r, _, info = env.step(ACTIONS.index(action_name))
        invalid_checks.append(info["valid_move"] is False and r == -10)
        env.reset()
    lines.append(f"Up/Down/Left from Start all correctly invalid: {all(invalid_checks)}")
    ok = ok and all(invalid_checks)

    q_path, q_reward, q_steps, q_goal = run_qlearning(
        env, episodes=300, epsilon=0.3, epsilon_decay=0.99, seed=3)
    lines.append(f"Q-learning greedy reward={q_reward} (expected 82), "
                  f"steps={q_steps}, goal reached={q_goal}")
    ok = ok and q_goal and (q_reward == 82)
 
    record(name, ok, lines)

def scenario_trap_avoidance():
    name = "Scenario 4: 3x3 grid with trap cells (tests avoidance behaviour)"
    grid = [
        [0, -10, 3],
        [2, -10, 4],
        [1, 2, 0],
    ]
    start, goal = (0, 0), (2, 2)
    trap_cells = {(0, 1), (1, 1)}
    env = Grid(grid, start, goal)
 
    path, reward, steps, reached_goal = trace_actions(
        env, ["Down", "Down", "Right", "Right"])
    lines = [f"hand-picked path reward = {reward} (expected 85), steps={steps}"]
    ok = reached_goal and reward == 85 and steps == 4
 
    bf_reward, _ = brute_force_optimal(env)
    matches = bf_reward == 85
    lines.append(f"brute-force optimal reward = {bf_reward} "
                  f"({'matches hand calc' if matches else 'DOES NOT MATCH hand calc'})")
    ok = ok and matches
    q_path, q_reward, q_steps, q_goal = run_qlearning(
        env, episodes=400, epsilon=0.3, epsilon_decay=0.99, seed=7)
    hit_trap = any(p in trap_cells for p in q_path)
    lines.append(f"Q-learning greedy path={q_path}")
    lines.append(f"Q-learning greedy reward={q_reward} (expected 85), "
                  f"trap cell visited={hit_trap}, goal reached={q_goal}")
    ok = ok and q_goal and (not hit_trap) and (q_reward == 85)
 
    record(name, ok, lines)

def main():
    print("=== Assignment 2: scenario-based verification ===\n")
    scenario_pdf_example()
    scenario_trivial_2x2()
    scenario_forced_corridor()
    scenario_trap_avoidance()
    n_pass = sum(1 for _, status in results if status == PASS)
    n_total = len(results)
    print("=" * 60)
    print(f"{n_pass}/{n_total} scenarios passed")
    if n_pass < n_total:
        for scenario_name, status in results:
            if status == FAIL:
                print(f"  FAILED: {scenario_name}")
        sys.exit(1)
 
if __name__ == "__main__":
    main()