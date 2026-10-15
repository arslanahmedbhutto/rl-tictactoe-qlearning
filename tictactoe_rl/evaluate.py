"""Measure an agent against opponents and against perfect play."""

import random

from tictactoe_rl.game import EMPTY, is_over, optimal_moves, play, reachable_states, to_move, winner

_STATES = None


def play_game(x_player, o_player, rng) -> int:
    """Returns the winner (1 = X, -1 = O, 0 = draw)."""
    board = EMPTY
    while not is_over(board):
        p = to_move(board)
        player = x_player if p == 1 else o_player
        board = play(board, player.choose(board, rng), p)
    return winner(board)


def match(agent, opponent, games: int = 500, seed: int = 0) -> dict:
    """Agent plays half the games as X and half as O. Results from the agent's side."""
    rng = random.Random(seed)
    out = {"X": {"win": 0, "draw": 0, "loss": 0}, "O": {"win": 0, "draw": 0, "loss": 0}}
    for g in range(games):
        side = "X" if g % 2 == 0 else "O"
        w = play_game(agent, opponent, rng) if side == "X" else play_game(opponent, agent, rng)
        mine = 1 if side == "X" else -1
        out[side]["win" if w == mine else "loss" if w == -mine else "draw"] += 1
    total = {k: out["X"][k] + out["O"][k] for k in ("win", "draw", "loss")}
    return {"as_X": out["X"], "as_O": out["O"], "total": total, "games": games}


def optimal_move_rate(agent) -> float:
    """Share of all 4,520 reachable positions where the agent's greedy move is minimax-optimal."""
    global _STATES
    _STATES = _STATES or reachable_states()
    rng = random.Random(0)
    good = sum(agent.choose(b, rng) in optimal_moves(b) for b in _STATES)
    return good / len(_STATES)
