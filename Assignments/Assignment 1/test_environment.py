"""
pytest unit tests for the Grid Environment, in isolation from agent
"""
import pytest
from grid import Grid, ACTIONS

"""
A small 3x3 grid with single obstacle, used for the tests below
"""

def make_env():
    return Grid(n=3, m=3, st=(0, 0), en=(2, 2), obs=[(1, 1)])

def test_step_valid_move():
    env = make_env()
    next_state, reward, done = env.step((0, 0), ACTIONS.index("Right"))
    assert next_state == (0, 1)
    assert reward == -1
    assert done is False

def test_step_out_of_bounds_is_invalid():
    env = make_env()
    next_state, reward, done = env.step((0, 0), ACTIONS.index("Up"))
    assert next_state == (0, 0)
    assert reward == -10
    assert done is False

def test_step_into_obstacle():
    env = make_env()
    next_state, reward, done = env.step((0, 1), ACTIONS.index("Down"))
    assert next_state == (0, 1)
    assert reward == -100
    assert done is False

def test_step_into_goal_ends_episode():
    env = make_env()
    next_state, reward, done = env.step((1, 2), ACTIONS.index("Down"))
    assert next_state == (2, 2)
    assert reward == 100
    assert done is True

def test_print_grid_shows_markers():
    env = make_env()
    lines = env.print_grid().splitlines()
    assert lines[0].split()[0] == "S"
    assert lines[2].split()[2] == "G"
    assert lines[1].split()[1] == "X"

def test_invalid_start_or_goal_raises():
    with pytest.raises(ValueError):
        Grid(n=3, m=3, st=(5, 5), en=(2, 2), obs=[])
    with pytest.raises(ValueError):
        Grid(n=3, m=3, st=(0, 0), en=(0, 0), obs=[])

def test_start_or_goal_on_obstacle_raises():
    with pytest.raises(ValueError):
        Grid(n=3, m=3, st=(0, 0), en=(2, 2), obs=[(0, 0)])
