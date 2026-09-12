from agent import Agent

def test_greedy_action_picks_max_q():
    """check if the agent is exploiting when epsilon is set 0.0"""
    agent = Agent(n_states=4, n_actions=4, eps=0.0, seed=0)
    agent.q[0] = [1.0, 5.0, -2.0, 0.0]
    assert agent.choose_action(0) == 1

def test_fully_random_when_epsilon_one():
    """when epsilon=1, the agent should explore all the options available"""
    agent = Agent(n_states=4, n_actions=4, eps=1.0, seed=1)
    agent.q[0] = [1.0, 5.0, -2.0, 0.0]
    seen = {agent.choose_action(0) for _ in range(200)}
    assert len(seen) > 1

def test_update_moves_q_towards_target():
    """checking if Bellman formula is followed"""
    agent = Agent(n_states=2, n_actions=2, alpha=0.5, gamma=0.9)
    agent.q[1] = [10.0, 0.0]
    before = agent.q[0, 0]
    agent.update(state=0, action=0, reward=1.0, next_state=1, done=False)
    after = agent.q[0, 0]
    target = 1.0 + 0.9 * 10.0
    assert after == before + 0.5 * (target - before)

def test_update_ignores_next_state_when_done():
    """when done=True, there should be no next step to move"""
    agent = Agent(n_states=2, n_actions=2, alpha=1.0, gamma=0.9)
    agent.q[1] = [999.0, 999.0]
    agent.update(state=0, action=0, reward=5.0, next_state=1, done=True)
    assert agent.q[0, 0] == 5.0

def test_epsilon_decay_respects_floor():
    """epsilon should never go less than defined floor"""
    agent = Agent(n_states=2, n_actions=2, eps=1.0, eps_decay=0.5, eps_min=0.2)
    for _ in range(10):
        agent.decay_eps()
    assert agent.eps == 0.2

def test_policy_grid_shape_and_symbols():
    """checking the integrity of the to be printed grid"""
    agent = Agent(n_states=6, n_actions=4, seed=0)
    grid = agent.policy_grid(2, 3)
    assert len(grid) == 2
    assert len(grid[0]) == 3
    allowed = {"^", "v", "<", ">"}
    for row in grid:
        for sym in row:
            assert sym in allowed