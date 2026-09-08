"""
pytest unit tests for the Q-Learning agent, in siolation from training 
"""
from grid import Grid
from agent import Agent

"""
A small 2x2 grid, used for the tests below
"""

def make_env():
    return Grid(n=2, m=2, st=(0, 0), en=(1, 1), obs=[])

def test_greedy_action_picks_max_q():
    agent = Agent(env=make_env(), alpha=0.1, gamma=0.9, eps=0.0, seed=0)
    agent.q[0, 0] = [1.0, 5.0, -2.0, 0.0]
    assert agent.choose_action((0, 0)) == 1

def test_fully_random_when_epsilon_one():
    agent = Agent(env=make_env(), alpha=0.1, gamma=0.9, eps=1.0, seed=1)
    agent.q[0, 0] = [1.0, 5.0, -2.0, 0.0]
    seen = {agent.choose_action((0, 0)) for _ in range(200)}
    assert len(seen) > 1

def test_greedy_flag_ignores_epsilon():
    agent = Agent(env=make_env(), alpha=0.1, gamma=0.9, eps=1.0, seed=2)
    agent.q[0, 0] = [1.0, 5.0, -2.0, 0.0]
    assert agent.choose_action((0, 0), greedy=True) == 1

def test_update_moves_q_towards_target():
    agent = Agent(env=make_env(), alpha=0.5, gamma=0.9, eps=0.0)
    agent.q[1, 1] = [10.0, 0.0, 0.0, 0.0]
    before = agent.q[0, 0, 0]
    agent.update(state=(0, 0), action=0, reward=1.0, next_state=(1, 1), done=False)
    after = agent.q[0, 0, 0]
    target = 1.0 + 0.9 * 10.0
    assert after == before + 0.5 * (target - before)

def test_update_ignores_next_state_when_done():
    agent = Agent(env=make_env(), alpha=1.0, gamma=0.9, eps=0.0)
    agent.q[1, 1] = [999.0, 999.0, 999.0, 999.0]
    agent.update(state=(0, 0), action=0, reward=5.0, next_state=(1, 1), done=True)
    assert agent.q[0, 0, 0] == 5.0

def test_epsilon_decay_respects_floor():
    agent = Agent(env=make_env(), alpha=0.1, gamma=0.9, eps=1.0, eps_decay=0.5, eps_min=0.2)
    for _ in range(10):
        agent.decay_eps()
    assert agent.eps == 0.2
