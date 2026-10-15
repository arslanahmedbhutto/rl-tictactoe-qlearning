"""Tabular Q-learning agent with two training regimes.

The Q-table is keyed by the board *from the mover's point of view* (own marks = 1),
so the same table can play X or O.
"""

import pickle
import random
from collections import defaultdict

from tictactoe_rl.game import EMPTY, canonical, legal_moves, play, to_move, winner


class QAgent:
    name = "Q-learning agent"

    def __init__(self, alpha: float = 0.3, gamma: float = 0.95):
        self.q = defaultdict(float)          # (canonical_board, move) -> value
        self.alpha, self.gamma = alpha, gamma

    # ---------- acting ----------
    def values(self, board) -> dict[int, float]:
        s = canonical(board, to_move(board))
        return {m: self.q[(s, m)] for m in legal_moves(board)}

    def choose(self, board, rng=random, epsilon: float = 0.0):
        moves = legal_moves(board)
        if rng.random() < epsilon:
            return rng.choice(moves)
        vals = self.values(board)
        best = max(vals.values())
        return rng.choice([m for m, v in vals.items() if v == best])

    # ---------- learning ----------
    def _target_max(self, board) -> float:
        return max(self.values(board).values())

    def train_self_play(self, episodes: int, epsilon: tuple[float, float] = (1.0, 0.05), rng=None, callback=None):
        """Both sides use and update the same table (negamax-style target:
        my value after a move = minus the opponent's best value)."""
        rng = rng or random.Random(0)
        for ep in range(episodes):
            eps = epsilon[0] + (epsilon[1] - epsilon[0]) * ep / max(1, episodes - 1)
            board = EMPTY
            while True:
                p = to_move(board)
                s = canonical(board, p)
                m = self.choose(board, rng, eps)
                nb = play(board, m, p)
                w = winner(nb)
                if w or 0 not in nb:
                    target = 1.0 if w else 0.0
                else:
                    target = -self.gamma * self._target_max(nb)
                self.q[(s, m)] += self.alpha * (target - self.q[(s, m)])
                if w or 0 not in nb:
                    break
                board = nb
            if callback:
                callback(ep + 1, self)

    def train_vs(self, opponent, episodes: int, epsilon: tuple[float, float] = (1.0, 0.05), rng=None, callback=None):
        """Standard Q-learning against a fixed opponent; the agent alternates between X and O."""
        rng = rng or random.Random(0)
        for ep in range(episodes):
            eps = epsilon[0] + (epsilon[1] - epsilon[0]) * ep / max(1, episodes - 1)
            me = 1 if ep % 2 == 0 else -1
            board, prev = EMPTY, None            # prev = (state, move) waiting for its target
            while True:
                if to_move(board) == me:
                    s = canonical(board, me)
                    if prev:                      # non-terminal transition: bootstrap
                        self._update(prev, self.gamma * self._target_max(board))
                    m = self.choose(board, rng, eps)
                    prev = (s, m)
                else:
                    m = opponent.choose(board, rng)
                board = play(board, m, to_move(board))
                w = winner(board)
                if w or 0 not in board:
                    self._update(prev, 1.0 if w == me else -1.0 if w else 0.0)
                    break
            if callback:
                callback(ep + 1, self)

    def _update(self, sa, target):
        self.q[sa] += self.alpha * (target - self.q[sa])

    # ---------- persistence ----------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"q": {k: v for k, v in self.q.items() if v != 0.0}, "alpha": self.alpha, "gamma": self.gamma}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        agent = cls(d["alpha"], d["gamma"])
        agent.q.update(d["q"])
        return agent

