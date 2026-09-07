import pytest
from grid import Grid, ACTIONS
from agent import Agent

def make_env():
    grid = [
        [0, 1, 2],
        [-1, 5, 3],
        [2, -4, 0],
    ]
    return Grid(grid_vals=grid, st=(0, 0), en=(2, 2))

def test_reset_state():
    env = make_env()
    s = env.reset()
    assert s == env.state_idx((0, 0))
    assert env.curr_pos == (0, 0)
    assert env.visited == {(0, 0)}
    assert env.steps_taken == 0
    
def test_out_of_bounds_is_invalid():
    env = make_env()
    env.reset()
    up_idx = ACTIONS.index("Up")  
    _, reward, done, info = env.step(up_idx)
    assert info["valid_move"] is False
    assert reward == -10
    assert env.curr_pos == (0, 0)
    assert done is False

def test_valid_move_collects_cell_value():
    env = make_env()
    env.reset()
    right_idx = ACTIONS.index("Right")
    _, reward, done, info = env.step(right_idx)
    assert info["valid_move"] is True
    assert reward == -5 + 1
    assert env.curr_pos == (0, 1)
    assert (0, 1) in env.visited

def test_revisit_is_invalid():
    env = make_env()
    env.reset()
    right_idx = ACTIONS.index("Right")
    left_idx = ACTIONS.index("Left")
    env.step(right_idx)  
    _, reward, done, info = env.step(left_idx)  
    assert info["valid_move"] is False
    assert reward == -10
    assert env.curr_pos == (0, 1)
 
 
def test_reaching_goal_gives_bonus_and_ends_episode():
    grid = [[0, 0], [0, 0]]
    env = Grid(grid, st=(0, 0), en=(0, 1))
    env.reset()
    right_idx = ACTIONS.index("Right")
    _, reward, done, info = env.step(right_idx)
    assert info["reached_goal"] is True
    assert reward == -5 + 100
    assert done is True
 
def test_episode_truncates_at_max_steps():
    grid = [[0, 1], [1, 0]]
    env = Grid(grid, st=(0, 0), en=(1, 1))
    assert env.max_steps == 2 * 2 * 2  
    env.reset()
    up_idx = ACTIONS.index("Up")  
    done, steps, info = False, 0, None
    while not done and steps < env.max_steps:
        _, reward, done, info = env.step(up_idx)
        steps += 1
    assert steps == env.max_steps
    assert done is True
    assert info["truncated"] is True
    assert info["reached_goal"] is False
 
def test_available_actions_excludes_visited_and_out_of_bounds():
    env = make_env()
    env.reset()
    avail = env.available_actions()
    names = {ACTIONS[a] for a in avail}
    assert "Up" not in names     
    assert "Left" not in names   
    assert "Right" in names
    assert "Down" in names
 
def test_invalid_start_or_goal_raises():
    grid = [[0, 1], [1, 0]]
    with pytest.raises(ValueError):
        Grid(grid, st=(0, 0), en=(0, 0))
    with pytest.raises(ValueError):
        Grid(grid, st=(5, 5), en=(0, 0))