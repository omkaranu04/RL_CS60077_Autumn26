import numpy as np

ACTIONS = ["Up", "Down", "Left", "Right"]
ACTION_DELTAS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}

R_GOAL_BONUS = 100
R_STEP = -5
R_INVALID = -10

class Grid:
    def __init__(self, grid_vals, st, en):
        self.grid_vals = np.array(grid_vals)
        self.n, self.m = self.grid_vals.shape
        self.st = tuple(st)
        self.en = tuple(en)
        self.max_steps = 2 * self.n * self.m
        self._validate()
        self.reset()

    def _validate(self):
        x, y = self.st
        if not (0 <= x < self.n and 0 <= y < self.m):
            raise ValueError(f"Start position {self.st} is outside the grid")
        x, y = self.en
        if not (0 <= x < self.n and 0 <= y < self.m):
            raise ValueError(f"Goal position {self.en} is outside the grid")
        if self.st == self.en:
            raise ValueError(f"Start and Goal cannot be the same cell")

    def state_idx(self, pos):
        x, y = pos
        return x * self.m + y

    @property
    def n_states(self):
        return self.n * self.m

    @property
    def n_actions(self):
        return len(ACTIONS)

    def reset(self):
        self.curr_pos = self.st
        self.visited = {self.st}
        self.steps_taken = 0
        self.path = [self.st]
        return self.state_idx(self.curr_pos)

    def check(self, pos):
        x, y = pos
        return 0 <= x < self.n and 0 <= y < self.m

    def step(self, action_idx):
        if self.steps_taken >= self.max_steps:
            raise RuntimeError("Episodes already finished (max steps reached)")

        action_name = ACTIONS[action_idx]
        dx, dy = ACTION_DELTAS[action_name]
        x, y = self.curr_pos
        new_pos = (x + dx, y + dy)
        valid = self.check(new_pos) and (new_pos not in self.visited)
        if not valid:
            reward = R_INVALID
            reached_goal = False
        else:
            if new_pos == self.en:
                reward = R_STEP + R_GOAL_BONUS
                reached_goal = True
            else:
                reward = R_STEP + int(self.grid_vals[new_pos])
                reached_goal = False
            self.curr_pos = new_pos
            self.visited.add(new_pos)
            self.path.append(new_pos)

        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_steps
        done = reached_goal or truncated

        info = {
            "valid_move": valid,
            "reached_goal": reached_goal,
            "truncated": truncated and not reached_goal,
            "action_name": action_name,
            "position_name": self.curr_pos,
        }
        return self.state_idx(self.curr_pos), reward, done, info

    def available_actions(self, pos=None):
        if pos is None:
            pos = self.curr_pos
        avail = []
        for idx, name in enumerate(ACTIONS):
            dx, dy = ACTION_DELTAS[name]
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self.check(new_pos) and new_pos not in self.visited:
                avail.append(idx)
        return avail
