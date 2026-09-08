# Action Space (word + directions + indexes)
ACTIONS = ["Up", "Down", "Left", "Right"]
ACTION_DELTAS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}

# Rewards as specified in the Assignment
R_GOAL = 100
R_STEP = -1
R_INVALID = -10
R_OBSTACLE = -100

class Grid:
    def __init__(self, n, m, st, en, obs):
        self.n = n                              # number of rows
        self.m = m                              # number of columns
        self.st = tuple(st)                     # start position
        self.en = tuple(en)                     # goal position
        self.obs = set(tuple(o) for o in obs)   # obstacle positions
        self.max_steps = 2 * n * m              # max. no. of steps per episode
        self._validate()

    def _validate(self):
        """
            Validating if the entered configurations are right before the training occurs
        """
        if not self.check(self.st):
            raise ValueError(f"Start position {self.st} is outside the grid")
        if not self.check(self.en):
            raise ValueError(f"Goal position {self.en} is outside the grid")
        if self.st == self.en:
            raise ValueError(f"Start and Goal cannot be the same cell")
        if self.st in self.obs or self.en in self.obs:
            raise ValueError("Start/Goal position cannot be an obstacle")
        for o in self.obs:
            if not self.check(o):
                raise ValueError(f"Obstacle {o} is outside the grid")

    def check(self, pos):
        """
            Checking 'pos' is within the grid limits
        """
        r, c = pos
        return 0 <= r < self.n and 0 <= c < self.m

    def step(self, state, action_idx):
        """
            Apply action using 'action_idx' while in state 'state'
            Returns: (next_state, reward, done)
                - move outside grid     -> stay in place, reward = -10
                - move into obstacle    -> stay in place, reward = -100
                - move into goal        -> episode ends,  reward = +100
                - any other move        -> valid move,    reward = -1
        """
        x, y = state
        action_name = ACTIONS[action_idx]
        dx, dy = ACTION_DELTAS[action_name]
        next_state = (x + dx, y + dy)
        if not self.check(next_state):
            return state, R_INVALID, False
        if next_state in self.obs:
            return state, R_OBSTACLE, False
        if next_state == self.en:
            return next_state, R_GOAL, True
        return next_state, R_STEP, False

    def print_grid(self):
        """
            Print the grid using symbols:
                S = start, G = goal, X = obstacle, . = empty
        """
        lines = []
        for x in range(self.n):
            row = []
            for y in range(self.m):
                pos = (x, y)
                if pos == self.st:
                    row.append("S")
                elif pos == self.en:
                    row.append("G")
                elif pos in self.obs:
                    row.append("X")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)
