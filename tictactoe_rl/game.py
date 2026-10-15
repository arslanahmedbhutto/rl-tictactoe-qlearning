"""Tic-Tac-Toe rules and a perfect (minimax) player.

Boards are tuples of 9 ints: 1 = X, -1 = O, 0 = empty. X always moves first.
"""

import random
from functools import lru_cache

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
EMPTY = (0,) * 9


def winner(board) -> int:
    """1 or -1 if that player has three in a row, else 0."""
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def legal_moves(board) -> list[int]:
    return [i for i, v in enumerate(board) if v == 0]


def to_move(board) -> int:
    """X (1) moves when both players have the same number of marks."""
    return 1 if sum(1 for v in board if v != 0) % 2 == 0 else -1


def is_over(board) -> bool:
    return winner(board) != 0 or 0 not in board


def play(board, square: int, player: int) -> tuple:
    assert board[square] == 0, "square taken"
    b = list(board)
    b[square] = player
    return tuple(b)


def canonical(board, player: int) -> tuple:
    """Board from `player`'s point of view: own marks = 1, opponent = -1.
    Lets one Q-table play both X and O."""
    return tuple(v * player for v in board)


@lru_cache(maxsize=None)
def minimax_value(board) -> int:
    """Game value with perfect play, from X's point of view: 1 win, 0 draw, -1 loss."""
    w = winner(board)
    if w or 0 not in board:
        return w
    p = to_move(board)
    values = [minimax_value(play(board, m, p)) for m in legal_moves(board)]
    return max(values) if p == 1 else min(values)


def optimal_moves(board) -> list[int]:
    """Every move that keeps the best achievable result for the player to move."""
    p = to_move(board)
    scored = {m: minimax_value(play(board, m, p)) * p for m in legal_moves(board)}
    best = max(scored.values())
    return [m for m, v in scored.items() if v == best]


def reachable_states() -> list[tuple]:
    """All non-terminal positions reachable from the empty board (4,520)."""
    seen, stack = {EMPTY}, [EMPTY]
    while stack:
        b = stack.pop()
        if is_over(b):
            continue
        p = to_move(b)
        for m in legal_moves(b):
            nb = play(b, m, p)
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return [b for b in seen if not is_over(b)]


class RandomPlayer:
    name = "Random"

    def choose(self, board, rng=random):
        return rng.choice(legal_moves(board))


class MinimaxPlayer:
    """Perfect play; picks randomly among equally good moves so games vary."""

    name = "Minimax (perfect)"

    def choose(self, board, rng=random):
        return rng.choice(optimal_moves(board))
