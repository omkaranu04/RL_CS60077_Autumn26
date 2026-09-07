import sys
from collections import deque
from grid import Grid
from agent import Agent
import utils

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

TEST_CASES = [
    {
        "name": "3x3, no obstacles",
        "n": 3, "m": 3,
        "start": (0, 0), "goal": (2, 2),
        "obstacles": [],
        "alpha": 0.1, "gamma": 0.9, "eps": 0.2, "episodes": 300,
    },
    {
        "name": "Assignment example grid (6x5)",
        "n": 6, "m": 5,
        "start": (0, 0), "goal": (5, 4),
        "obstacles": [(0, 4), (1, 2), (2, 0), (2, 4), (3, 3), (4, 0), (5, 2)],
        "alpha": 0.1, "gamma": 0.9, "eps": 0.2, "episodes": 1000,
    },
    {
        "name": "5x5, forced diagonal detour",
        "n": 5, "m": 5,
        "start": (0, 0), "goal": (4, 4),
        "obstacles": [(1, 1), (2, 2), (4, 1), (3, 4), (3, 3)],
        "alpha": 0.1, "gamma": 0.3, "eps": 0.2, "episodes": 1000,
    },
    {
        "name": "5x5, narrow corridor (forced detour)",
        "n": 5, "m": 5,
        "start": (0, 0), "goal": (0, 4),
        "obstacles": [(0, 1), (0, 2), (0, 3)],
        "alpha": 0.1, "gamma": 0.9, "eps": 0.3, "episodes": 800,
    },
    {
        "name": "8x8, sparse obstacles",
        "n": 8, "m": 8,
        "start": (0, 0), "goal": (7, 7),
        "obstacles": [(1, 3), (2, 3), (3, 3), (4, 3), (5, 5), (6, 1), (0, 6), (7, 2)],
        "alpha": 0.15, "gamma": 0.9, "eps": 0.25, "episodes": 4000,
    },
    {
        "name": "Invalid config: obstacle on start cell",
        "n": 3, "m": 3,
        "start": (0, 0), "goal": (2, 2),
        "obstacles": [(0, 0)],
        "alpha": 0.1, "gamma": 0.9, "eps": 0.2, "episodes": 100,
        "expect_invalid": True,
    },
]

# Independent Ground Truth
def bfs(n, m, st, en, obs):
    obs = set(obs)
    if st == en:
        return 0
    visited = {st}
    q = deque([(st, 0)])
    while q:
        (x, y), d = q.popleft()
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < n and 0 <= ny < m):
                continue
            if (nx, ny) in obs or (nx, ny) in visited:
                continue
            if (nx, ny) == en:
                return d + 1
            visited.add((nx, ny))
            q.append(((nx, ny), d + 1))
    return None

PASS, FAIL = "PASS", "FAIL"
results = []

def record(name, ok, lines):
    results.append((name, PASS if ok else FAIL))
    print(f"[{PASS if ok else FAIL}] {name}")
    for line in lines:
        print(f"       {line}")
    print()

def verify_case(case):
    name = case["name"]
    lines = [f"Grid: {case['n']}x{case['m']}   Start: {case['start']}   Goal: {case['goal']}   "
             f"Obstacles: {len(case['obstacles'])}   Episodes: {case['episodes']}"]

    try:
        env = Grid(n=case["n"], m=case["m"], st=case["start"], en=case["goal"], obs=case["obstacles"])
    except ValueError as e:
        if case.get("expect_invalid"):
            lines.append(f"invalid config correctly rejected: {e}")
            record(name, True, lines)
            return
        lines.append(f"FAIL - unexpected ValueError: {e}")
        record(name, False, lines)
        return

    if case.get("expect_invalid"):
        lines.append("FAIL - expected a ValueError but the environment was accepted")
        record(name, False, lines)
        return

    optimal_len = bfs(case["n"], case["m"], case["start"], case["goal"], case["obstacles"])
    lines.append(f"True shortest path length (BFS): {optimal_len}")

    agent = Agent(env=env, alpha=case["alpha"], gamma=case["gamma"], eps=case["eps"], seed=0)
    utils.train(env=env, agent=agent, episodes=case["episodes"])
    path, actions, step_rewards, total_reward, success = utils.get_best_path(env=env, agent=agent)

    if not success:
        lines.append("FAIL - agent's greedy policy did not reach the goal")
        record(name, False, lines)
        return

    learned_len = len(actions)
    expected_reward = (learned_len - 1) * (-1) + 100
    lines.append(f"Learned path length: {learned_len}    Total reward: {total_reward}")

    ok = True
    if learned_len != optimal_len:
        lines.append(f"FAIL - learned path length ({learned_len}) != optimal ({optimal_len})")
        ok = False
    if total_reward != expected_reward:
        lines.append(f"FAIL - total reward ({total_reward}) != expected ({expected_reward})")
        ok = False

    record(name, ok, lines)

def main():
    print("=== Assignment 1: scenario-based verification ===\n")
    for case in TEST_CASES:
        verify_case(case)

    n_pass = sum(1 for _, status in results if status == PASS)
    n_total = len(results)
    print("=" * 60)
    print(f"{n_pass}/{n_total} scenarios passed")
    if n_pass < n_total:
        for name, status in results:
            if status == FAIL:
                print(f"  FAILED: {name}")
        sys.exit(1)

if __name__ == "__main__":
    main()
