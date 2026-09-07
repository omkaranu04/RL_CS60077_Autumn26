import numpy as np

EMPTY, X, O = 0, 1, 2
SYMBOLS = {EMPTY: ".", X: "X", O: "O"}

R_WIN = 100
R_STEP = -1
R_INVALID = -10
R_LOSE = -100
R_DRAW = 0

class Board:
    def __init__(self, size=3):
        self.size = size
        self.n_cells = size * size
        self.max_steps = 10 * self.n_cells
        self._lines = self._winning_lines()
        self.reset()

    def _winning_lines(self):
        lines = []
        for r in range(self.size):
            lines.append([r * self.size + c for c in range(self.size)])
        for c in range(self.size):
            lines.append([r * self.size + c for r in range(self.size)])
        lines.append([i * self.size + i for i in range(self.size)])
        lines.append([i * self.size + (self.size - 1 - i) for i in range(self.size)])
        return lines

    @property
    def n_states(self):
        return 3 ** self.n_cells

    @property
    def n_actions(self):
        return self.n_cells

    def reset(self):
        self.cells = [EMPTY] * self.n_cells
        self.current_player = X
        self.steps_taken = 0
        return self.state_idx(tuple(self.cells))

    def state_idx(self, state):
        idx = 0
        for cell in state:
            idx = idx * 3 + cell
        return idx

    def state_from_idx(self, idx):
        cells = [EMPTY] * self.n_cells
        for i in range(self.n_cells - 1, -1, -1):
            cells[i] = idx % 3
            idx //= 3
        return tuple(cells)

    def check(self, action):
        return 0 <= action < self.n_cells

    def available_actions(self, state=None):
        cells = state if state is not None else self.cells
        return [i for i, v in enumerate(cells) if v == EMPTY]

    def winner(self, cells=None):
        cells = cells if cells is not None else self.cells
        for line in self._lines:
            vals = [cells[i] for i in line]
            if vals[0] != EMPTY and all(v == vals[0] for v in vals):
                return vals[0]
        return None

    def step(self, action):
        """
            Apply 'action' (cell index) for the current player.
            Returns: (next_state_idx, reward, done, info)
            Rules:
                - mark an occupied cell   -> stay in place, reward = -10, same player's turn
                - mark that wins          -> episode ends,  reward = +100
                - mark that fills board,
                  no winner (draw)        -> episode ends,  reward = 0
                - any other mark          -> valid move,    reward = -1, turn passes
            The opponent's own win/lose/draw reward (+100/-100/0) is credited
            to their last move by the training loop in utils.play_episode,
            since a single call to step() only knows the mover's outcome.
        """
        if self.steps_taken >= self.max_steps:
            raise RuntimeError("Episode already finished (max steps reached)")
        self.steps_taken += 1

        player = self.current_player
        if not self.check(action) or self.cells[action] != EMPTY:
            info = {"valid_move": False, "winner": None, "draw": False, "player": player}
            return self.state_idx(tuple(self.cells)), R_INVALID, False, info

        self.cells[action] = player
        win = self.winner()
        draw = (win is None) and (EMPTY not in self.cells)
        done = (win is not None) or draw

        if win is not None:
            reward = R_WIN
        elif draw:
            reward = R_DRAW
        else:
            reward = R_STEP

        info = {"valid_move": True, "winner": win, "draw": draw, "player": player}
        next_state = tuple(self.cells)
        if not done:
            self.current_player = O if player == X else X

        return self.state_idx(next_state), reward, done, info

    def print_board(self, cells=None):
        cells = cells if cells is not None else self.cells
        rows = []
        for r in range(self.size):
            row = cells[r * self.size:(r + 1) * self.size]
            rows.append(" ".join(SYMBOLS[c] for c in row))
        return "\n".join(rows)
