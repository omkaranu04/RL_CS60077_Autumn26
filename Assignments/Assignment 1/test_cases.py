import subprocess
import sys
from collections import deque
PYTHON = sys.executable

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

def build_input(case):
    lines = [str(case["n"]), str(case["m"])]
    lines.append(f"{case['start'][0]},{case['start'][1]}")
    lines.append(f"{case['goal'][0]},{case['goal'][1]}")
    lines.append(str(len(case["obstacles"])))
    for (r, c) in case["obstacles"]:
        lines.append(f"{r},{c}")
    lines.append(str(case["alpha"]))
    lines.append(str(case["gamma"]))
    lines.append(str(case["eps"]))
    lines.append(str(case["episodes"]))
    return "\n".join(lines) + "\n"

def run_main(case):
    proc = subprocess.run(
        [PYTHON, "main.py"],
        input=build_input(case),
        text=True,
        capture_output=True,
        timeout=300,
    )
    return proc.returncode, proc.stdout, proc.stderr

def parse_output(output):
    result = {"num_steps": None, "total_reward": None, "reached_goal": None}
    if "The policy did not reach the goal" in output:
        result["reached_goal"] = False
        return result
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Number of steps:"):
            result["num_steps"] = int(line.split(":")[1].strip())
        elif line.startswith("Total") and result["total_reward"] is None:
            parts = line.split()
            if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
                result["total_reward"] = int(parts[-1])
    result["reached_goal"] = result["num_steps"] is not None
    return result

def verify_case(case):
    print(f"--- {case['name']} ---")
    print(f"  Grid: {case['n']}x{case['m']}   Start: {case['start']}   Goal: {case['goal']}   "
          f"Obstacles: {len(case['obstacles'])}   Episodes: {case['episodes']}")

    returncode, stdout, stderr = run_main(case)

    if case.get("expect_invalid"):
        if returncode == 0 and "Invalid environment configuration" in stdout:
            print("  PASS - invalid config was rejected gracefully (no crash)")
            return True
        print("  FAIL - expected a graceful 'Invalid environment configuration' message")
        return False

    if returncode != 0:
        print("  FAIL - main.py crashed:")
        last_line = stderr.strip().splitlines()[-1] if stderr.strip() else "(no stderr captured)"
        print(f"    {last_line}")
        return False

    optimal_len = bfs(case["n"], case["m"], case["start"], case["goal"], case["obstacles"])
    print(f"  True shortest path length (BFS): {optimal_len}")

    result = parse_output(stdout)
    if not result["reached_goal"]:
        print("  FAIL - agent's greedy policy did not reach the goal")
        return False

    learned_len = result["num_steps"]
    expected_reward = (learned_len - 1) * (-1) + 100
    print(f"  Learned path length: {learned_len}    Total reward: {result['total_reward']}")

    ok = True
    if learned_len != optimal_len:
        print(f"  FAIL - learned path length ({learned_len}) != optimal ({optimal_len})")
        ok = False
    if result["total_reward"] != expected_reward:
        print(f"  FAIL - total reward ({result['total_reward']}) != expected ({expected_reward})")
        ok = False

    if ok:
        print("  PASS")
    return ok

def main():
    print("Running test cases for main.py...")
    results = []
    for case in TEST_CASES:
        ok = verify_case(case)
        results.append((case["name"], ok))
        print()
        
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} test cases passed.")
    
if __name__ == "__main__":
    main()