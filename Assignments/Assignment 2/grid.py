import numpy as np

# index -> action name
ACTIONS = ["Up", "Down", "Left", "Right"]
# action name -> delta
ACTION_DELTAS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}

# rewards
R_GOAL_BONUS = 100
R_STEP = -5
R_INVALID = -10

class Grid:
    """A single-agent grid MDP with hidden rewards and no cell revisits"""
    def __init__(self, grid_vals, st, en):
        self.grid_vals = np.array(grid_vals)
        self.n, self.m = self.grid_vals.shape
        self.st = tuple(st)
        self.en = tuple(en)
        self.max_steps = 2 * self.n * self.m
        self._validate()
        self.reset()

    def _validate(self):
        """
            All the possible sanity checks for the grind integrity
        """
        x, y = self.st
        if not (0 <= x < self.n and 0 <= y < self.m):
            raise ValueError(f"Start position {self.st} is outside the grid")
        x, y = self.en
        if not (0 <= x < self.n and 0 <= y < self.m):
            raise ValueError(f"Goal position {self.en} is outside the grid")
        if self.st == self.en:
            raise ValueError(f"Start and Goal cannot be the same cell")

    def state_idx(self, pos):
        """
            Indexing the (row, col) to flat
        """
        x, y = pos
        return x * self.m + y

    @property
    def n_states(self):
        """total grid cells = n * m"""
        return self.n * self.m

    @property
    def n_actions(self):
        """number of possible actions = 4"""
        return len(ACTIONS)

    def reset(self):
        """
            Start a new episode by resetting all the required variables
        """
        self.curr_pos = self.st
        self.visited = {self.st}
        self.steps_taken = 0
        self.path = [self.st]
        return self.state_idx(self.curr_pos)

    def check(self, pos):
        """bound check"""
        x, y = pos
        return 0 <= x < self.n and 0 <= y < self.m

    def step(self, action_idx):
        """
            Apply action at 'action_idx' index
            Return: next_state, reward, done, info (dict)
        """
        if self.steps_taken >= self.max_steps:
            raise RuntimeError("Episodes already finished (max steps reached)")

        # where does the action takes the agent
        action_name = ACTIONS[action_idx]
        dx, dy = ACTION_DELTAS[action_name]
        x, y = self.curr_pos
        new_pos = (x + dx, y + dy)
        # move only if valid
        valid = self.check(new_pos) and (new_pos not in self.visited)
        # rewards, reache -> decided based on where we land
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
        """
            Which of the 4 actions currently lead to valid move/cell
        """
        if pos is None:
            pos = self.curr_pos
        avail = []
        for idx, name in enumerate(ACTIONS):
            dx, dy = ACTION_DELTAS[name]
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self.check(new_pos) and new_pos not in self.visited:
                avail.append(idx)
        return avail
