import sys
from board import Board, X, O
from agent import Agent
import utils

class ScriptedAgent:
    """Replays a fixed list of actions in order, ignoring state/greedy/updates."""
    def __init__(self, moves):
        self.moves = list(moves)
        self.i = 0

    def choose_action(self, state, greedy=False):
        action = self.moves[self.i]
        self.i += 1
        return action

    def update(self, *args, **kwargs):
        pass

PASS, FAIL = "PASS", "FAIL"
results = []

def record(name, ok, lines):
    results.append((name, PASS if ok else FAIL))
    print(f"[{PASS if ok else FAIL}] {name}")
    for line in lines:
        print(f"       {line}")
    print()

def scenario_win_with_invalid_retry():
    name = "Scenario 1: hand-scripted win, with one invalid retry for the loser"
    board = Board(size=3)
    agent_x = ScriptedAgent([0, 1, 2])
    agent_o = ScriptedAgent([0, 3, 4])  # first move (0) is already X's -> invalid, retried as 3

    ep_reward, marks_placed, winner, draw, trace = utils.play_episode(
        board, agent_x, agent_o, learn=False, greedy=False, record_trace=True)

    ok = (winner == X and not draw and marks_placed == 5
          and ep_reward[X] == 98.0 and ep_reward[O] == -112.0)
    lines = [
        f"winner={winner}, draw={draw}, marks_placed={marks_placed}",
        f"reward_X={ep_reward[X]} (expected 98.0)   reward_O={ep_reward[O]} (expected -112.0)",
    ]
    record(name, ok, lines)

def scenario_draw():
    name = "Scenario 2: hand-scripted draw, symmetric reward for both agents"
    board = Board(size=3)
    agent_x = ScriptedAgent([0, 2, 3, 7, 8])
    agent_o = ScriptedAgent([1, 4, 5, 6])

    ep_reward, marks_placed, winner, draw, trace = utils.play_episode(
        board, agent_x, agent_o, learn=False, greedy=False, record_trace=True)

    ok = (winner is None and draw and marks_placed == 9
          and ep_reward[X] == -4.0 and ep_reward[O] == -4.0)
    lines = [
        f"winner={winner}, draw={draw}, marks_placed={marks_placed}",
        f"reward_X={ep_reward[X]} (expected -4.0)   reward_O={ep_reward[O]} (expected -4.0)",
    ]
    record(name, ok, lines)

def scenario_self_play_converges_to_draws():
    name = "Scenario 3: self-play training converges toward draws (optimal Tic-Tac-Toe)"
    board = Board(size=3)
    agent_x = Agent(board.n_states, board.n_actions, alpha=0.1, gamma=0.9, eps=0.3, eps_decay=0.9995, seed=42)
    agent_o = Agent(board.n_states, board.n_actions, alpha=0.1, gamma=0.9, eps=0.3, eps_decay=0.9995, seed=43)

    _, _, _, outcomes = utils.train_self_play(board, agent_x, agent_o, episodes=20000)
    last = outcomes[-500:]
    draw_rate = last.count("draw") / len(last)

    ep_reward, marks_placed, winner, draw, trace = utils.play_episode(
        board, agent_x, agent_o, learn=False, greedy=True, record_trace=True)

    ok = draw and draw_rate >= 0.4
    lines = [
        f"draw rate over last {len(last)} training episodes: {draw_rate:.1%} (expected >= 40%)",
        f"greedy self-play from empty board: {'draw' if draw else ('X wins' if winner == X else 'O wins')} "
        f"in {marks_placed} moves (expected: draw)",
    ]
    record(name, ok, lines)

def main():
    print("=== Assignment 3: scenario-based verification ===\n")
    scenario_win_with_invalid_retry()
    scenario_draw()
    scenario_self_play_converges_to_draws()

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
