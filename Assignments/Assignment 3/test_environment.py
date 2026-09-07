from board import Board, X, O, EMPTY

def make_board(size=3):
    return Board(size=size)

def test_reset_state():
    b = make_board()
    s = b.reset()
    assert s == b.state_idx((EMPTY,) * 9)
    assert b.cells == [EMPTY] * 9
    assert b.current_player == X
    assert b.steps_taken == 0

def test_valid_move_updates_cell_and_switches_player():
    b = make_board()
    b.reset()
    _, reward, done, info = b.step(0)
    assert info["valid_move"] is True
    assert reward == -1
    assert done is False
    assert b.cells[0] == X
    assert b.current_player == O

def test_move_into_occupied_cell_is_invalid():
    b = make_board()
    b.reset()
    b.step(0)  # X marks cell 0
    _, reward, done, info = b.step(0)  # O tries the same cell
    assert info["valid_move"] is False
    assert reward == -10
    assert done is False
    assert b.current_player == O  # turn did not pass
    assert b.cells[0] == X  # unchanged

def test_row_win_detected():
    b = make_board()
    b.reset()
    moves = [0, 3, 1, 4, 2]  # X: 0,1,2 (top row)  O: 3,4
    reward = done = info = None
    for m in moves:
        _, reward, done, info = b.step(m)
    assert done is True
    assert info["winner"] == X
    assert reward == 100

def test_draw_detected():
    b = make_board()
    b.reset()
    # Final layout (no winner):  X O X / X O O / O X X
    moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]
    reward = done = info = None
    for m in moves:
        _, reward, done, info = b.step(m)
    assert done is True
    assert info["winner"] is None
    assert info["draw"] is True
    assert reward == 0

def test_available_actions_excludes_occupied():
    b = make_board()
    b.reset()
    b.step(0)
    avail = b.available_actions()
    assert 0 not in avail
    assert len(avail) == 8

def test_state_idx_round_trip():
    b = make_board()
    b.reset()
    b.step(4)
    cells = tuple(b.cells)
    idx = b.state_idx(cells)
    assert b.state_from_idx(idx) == cells

def test_print_board_shows_marks():
    b = make_board()
    b.reset()
    b.step(0)
    lines = b.print_board().splitlines()
    assert lines[0].split()[0] == "X"
