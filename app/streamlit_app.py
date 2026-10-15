"""Play Tic-Tac-Toe against the Q-learning agent and see what it was thinking.

    streamlit run app/streamlit_app.py
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from tictactoe_rl.agent import QAgent
from tictactoe_rl.game import EMPTY, is_over, optimal_moves, play, to_move, winner

st.set_page_config(page_title="Q-learning Tic-Tac-Toe", page_icon="🎮", layout="wide")
# Keep the 3x3 board as a grid on narrow (phone) screens instead of stacking the columns
st.markdown("<style>[data-testid='stHorizontalBlock']{flex-wrap:nowrap !important;}"
            "[data-testid='stColumn']{min-width:0 !important;}</style>", unsafe_allow_html=True)
SYMBOL = {1: "❌", -1: "⭕", 0: " "}
AGENTS = {"Self-play agent (strong)": "self", "Agent trained vs random (weaker)": "random"}


@st.cache_resource
def load_agent(regime):
    return QAgent.load(ROOT / "models" / f"q_{regime}.pkl")


def new_game():
    st.session_state.board = EMPTY
    st.session_state.thoughts = None


def agent_move(agent):
    board = st.session_state.board
    vals = agent.values(board)
    move = agent.choose(board, random.Random())
    st.session_state.thoughts = {"board": board, "values": vals, "move": move, "optimal": optimal_moves(board)}
    st.session_state.board = play(board, move, to_move(board))


st.title("🎮 Play against a Q-learning agent")
st.write("The agent learned Tic-Tac-Toe from scratch with tabular Q-learning. Perfect play always ends in a draw: can you beat it?")

c1, c2, c3 = st.columns([2, 1.3, 1])
choice = c1.selectbox("Opponent", list(AGENTS))
human = c2.radio("You play", ["❌ X (first)", "⭕ O (second)"], horizontal=True)
c3.write("")
if c3.button("🔄 New game", use_container_width=True) or "board" not in st.session_state:
    new_game()

agent = load_agent(AGENTS[choice])
me = 1 if human.startswith("❌") else -1
if st.session_state.get("side") != me or st.session_state.get("agent_name") != choice:
    st.session_state.side, st.session_state.agent_name = me, choice
    new_game()

board = st.session_state.board
if not is_over(board) and to_move(board) != me:
    agent_move(agent)
    board = st.session_state.board

left, right = st.columns([1, 1.2])
with left:
    for r in range(3):
        cols = st.columns(3)
        for c in range(3):
            i = 3 * r + c
            disabled = board[i] != 0 or is_over(board) or to_move(board) != me
            if cols[c].button(SYMBOL[board[i]], key=f"cell{i}", disabled=disabled, use_container_width=True):
                st.session_state.board = play(board, i, me)
                if not is_over(st.session_state.board):
                    agent_move(agent)
                st.rerun()

    w = winner(board)
    if w == me:
        st.success("You won! 🎉 (This agent has a blind spot.)")
    elif w == -me:
        st.error("The agent won.")
    elif is_over(board):
        st.info("Draw: the best result possible against perfect play.")
    else:
        st.caption("Your turn." if to_move(board) == me else "Agent is thinking…")

with right:
    t = st.session_state.thoughts
    if t:
        st.subheader("What the agent was thinking")
        st.caption("Its learned Q-value for each legal move (higher = better for the agent). "
                   "Green outline = a move a perfect player would also make.")
        rows = []
        for r in range(3):
            cells = []
            for c in range(3):
                i = 3 * r + c
                if i in t["values"]:
                    v = t["values"][i]
                    border = "3px solid #1baf7a" if i in t["optimal"] else "1px solid #e4e3df"
                    bg = "#2a78d6" if i == t["move"] else "transparent"
                    fg = "white" if i == t["move"] else "inherit"
                    cells.append(f'<td style="border:{border};background:{bg};color:{fg};width:70px;height:56px;'
                                 f'text-align:center;font-weight:600">{v:+.2f}</td>')
                else:
                    cells.append(f'<td style="border:1px solid #e4e3df;width:70px;height:56px;text-align:center">'
                                 f'{SYMBOL[t["board"][i]]}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        st.markdown('<table style="border-collapse:collapse">' + "".join(rows) + "</table>", unsafe_allow_html=True)
        verdict = "a perfect move ✅" if t["move"] in t["optimal"] else "not a perfect move ❌"
        st.write(f"It played square {t['move'] + 1} (blue): {verdict}.")
