"""Train Q-learning agents with two regimes, track learning curves, evaluate, save.

    python -m tictactoe_rl.train
"""

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tictactoe_rl.agent import QAgent
from tictactoe_rl.evaluate import match, optimal_move_rate
from tictactoe_rl.game import MinimaxPlayer, RandomPlayer

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR, REPORTS_DIR = ROOT / "models", ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EPISODES = 200_000
CHECK_EVERY = 10_000

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
})

REGIMES = {
    "Trained vs random opponent": "random",
    "Trained by self-play": "self",
}


def checkpoint(curve):
    def cb(ep, agent):
        if ep % CHECK_EVERY == 0:
            vs_mm = match(agent, MinimaxPlayer(), games=200, seed=ep)["total"]
            vs_rnd = match(agent, RandomPlayer(), games=400, seed=ep)["total"]
            curve.append({"episode": ep, "optimal_rate": optimal_move_rate(agent),
                          "loss_vs_minimax": vs_mm["loss"] / 200, "loss_vs_random": vs_rnd["loss"] / 400,
                          "win_vs_random": vs_rnd["win"] / 400})
            print(f"   {ep:>7,}: optimal moves {curve[-1]['optimal_rate']:.1%} | loses to minimax "
                  f"{curve[-1]['loss_vs_minimax']:.1%} | beats random {curve[-1]['win_vs_random']:.1%}", flush=True)
    return cb


def main():
    results, curves = {}, {}
    for name, regime in REGIMES.items():
        print(f"== {name}", flush=True)
        agent, curve, t0 = QAgent(), [], time.time()
        rng = random.Random(42)
        if regime == "random":
            agent.train_vs(RandomPlayer(), EPISODES, rng=rng, callback=checkpoint(curve))
        else:
            agent.train_self_play(EPISODES, rng=rng, callback=checkpoint(curve))
        minutes = (time.time() - t0) / 60
        MODELS_DIR.mkdir(exist_ok=True)
        agent.save(MODELS_DIR / f"q_{regime}.pkl")
        results[name] = {
            "regime": regime, "episodes": EPISODES, "train_minutes": round(minutes, 1),
            "q_entries": sum(1 for v in agent.q.values() if v != 0),
            "optimal_move_rate": optimal_move_rate(agent),
            "vs_random": match(agent, RandomPlayer(), games=2000, seed=7),
            "vs_minimax": match(agent, MinimaxPlayer(), games=1000, seed=7),
        }
        curves[name] = curve

    baseline = {"Random player": {
        "vs_random": match(RandomPlayer(), RandomPlayer(), games=2000, seed=7),
        "vs_minimax": match(RandomPlayer(), MinimaxPlayer(), games=1000, seed=7),
    }}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps({
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results, "baseline": baseline, "curves": curves}, indent=2), encoding="utf-8")
    write_table(results, baseline)
    plot(results, curves, baseline)


def pct(d, k):
    return f"{100 * d['total'][k] / d['games']:.1f}%"


def write_table(results, baseline):
    lines = ["| Agent | Optimal moves | vs random: win / draw / loss | vs perfect player: draw / loss |",
             "|---|---|---|---|"]
    for name, r in {**results, **baseline}.items():
        opt = f"{100 * r['optimal_move_rate']:.1f}%" if "optimal_move_rate" in r else "—"
        lines.append(f"| {name} | {opt} | {pct(r['vs_random'], 'win')} / {pct(r['vs_random'], 'draw')} / "
                     f"{pct(r['vs_random'], 'loss')} | {pct(r['vs_minimax'], 'draw')} / {pct(r['vs_minimax'], 'loss')} |")
    md = "# Results\n\nGames alternate sides (half as X, half as O). A perfect player can never be beaten, so draws are the best possible result against it.\n\n" + "\n".join(lines) + "\n"
    (REPORTS_DIR / "results.md").write_text(md, encoding="utf-8")
    print(md)


def plot(results, curves, baseline):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for (name, curve), color in zip(curves.items(), [ORANGE, BLUE]):
        eps = [c["episode"] / 1000 for c in curve]
        axes[0].plot(eps, [100 * c["optimal_rate"] for c in curve], "-o", ms=4, color=color, lw=2, label=name)
        axes[1].plot(eps, [100 * c["loss_vs_minimax"] for c in curve], "-o", ms=4, color=color, lw=2, label=name)
    axes[0].set_title("Positions where the agent plays a perfect move")
    axes[0].set_ylabel("% of 4,520 positions")
    axes[0].set_ylim(0, 100)
    axes[1].set_title("Games lost to a perfect (minimax) player")
    axes[1].set_ylabel("% of games")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("Training episodes (thousands)")
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    names = list(results) + ["Random player"]
    data = {**results, **baseline}
    stacks = [("draw", AQUA, "Draw"), ("loss", RED, "Loss")]
    left = [0.0] * len(names)
    for key, color, label in stacks:
        vals = [100 * data[n]["vs_minimax"]["total"][key] / data[n]["vs_minimax"]["games"] for n in names]
        ax.barh(names[::-1], vals[::-1], left=left[::-1], color=color, label=label, height=0.55,
                edgecolor=SURFACE, linewidth=2)
        for i, (v, l) in enumerate(zip(vals[::-1], left[::-1])):
            if v >= 6:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of 1,000 games against a perfect player (draw is the best possible result)")
    ax.set_title("Can it hold a perfect player to a draw?")
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.1), frameon=False)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES_DIR / "vs_perfect_player.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
