import random
from pathlib import Path

import pytest

from tictactoe_rl.agent import QAgent
from tictactoe_rl.evaluate import match, optimal_move_rate
from tictactoe_rl.game import (EMPTY, MinimaxPlayer, RandomPlayer, canonical, is_over, legal_moves, minimax_value,
                               optimal_moves, play, reachable_states, to_move, winner)

MODELS = Path(__file__).resolve().parent.parent / "models"


def test_rules():
    b = EMPTY
    for sq, p in [(0, 1), (3, -1), (1, 1), (4, -1), (2, 1)]:
        assert to_move(b) == p
        b = play(b, sq, p)
    assert winner(b) == 1 and is_over(b)
    assert legal_moves(b) == [5, 6, 7, 8]
    with pytest.raises(AssertionError):
        play(b, 0, -1)


def test_draw_board():
    b = (1, -1, 1, 1, -1, -1, -1, 1, 1)
    assert winner(b) == 0 and is_over(b)


def test_state_space_and_game_value():
    assert len(reachable_states()) == 4520   # known count of non-terminal positions
    assert minimax_value(EMPTY) == 0         # perfect play is a draw


def test_optimal_moves_take_wins_and_block():
    win_now = (1, 1, 0, -1, -1, 0, 0, 0, 0)   # X to move: square 2 wins
    assert optimal_moves(win_now) == [2]
    block = (1, 0, 0, -1, -1, 0, 1, 0, 0)     # X to move: must block at 5
    assert optimal_moves(block) == [5]


def test_canonical_flips_perspective():
    b = (1, -1, 0, 0, 0, 0, 0, 0, 0)
    assert canonical(b, -1) == (-1, 1, 0, 0, 0, 0, 0, 0, 0)


def test_minimax_never_loses():
    res = match(MinimaxPlayer(), RandomPlayer(), games=100, seed=1)
    assert res["total"]["loss"] == 0
    assert match(MinimaxPlayer(), MinimaxPlayer(), games=20)["total"]["draw"] == 20


def test_short_self_play_learns_something():
    agent = QAgent()
    agent.train_self_play(3000, rng=random.Random(0))
    assert optimal_move_rate(agent) > 0.5
    assert match(agent, RandomPlayer(), games=200)["total"]["win"] > 100


def test_save_and_load_roundtrip(tmp_path):
    agent = QAgent()
    agent.train_vs(RandomPlayer(), 200, rng=random.Random(0))
    agent.save(tmp_path / "q.pkl")
    loaded = QAgent.load(tmp_path / "q.pkl")
    assert all(loaded.q[k] == v for k, v in agent.q.items() if v != 0)


def test_saved_self_play_agent_is_near_perfect():
    agent = QAgent.load(MODELS / "q_self.pkl")
    assert optimal_move_rate(agent) > 0.95
    assert match(agent, MinimaxPlayer(), games=200)["total"]["loss"] <= 10
