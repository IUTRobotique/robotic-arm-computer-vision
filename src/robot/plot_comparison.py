from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np

LOGS_DIR: Path = Path(__file__).parent / "logs"
OUT_DIR:  Path = Path(__file__).parent

ALGOS: dict[str, dict[str, Any]] = {
    "HER-SAC": {"path": "her_sac", "tb_prefix": "SAC",  "color": "#2196F3", "marker": "o"},
    "PPO":     {"path": "ppo",     "tb_prefix": "PPO",  "color": "#F44336", "marker": "s"},
    "TD3":     {"path": "td3",     "tb_prefix": "TD3",  "color": "#4CAF50", "marker": "^"},
}


# ── Chargement (inchangé) ──────────────────────────────────────────────────────

def _tb_run_steps(run_path: Path) -> int:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(run_path), size_guidance={"scalars": 0})
        ea.Reload()
        events = ea.Scalars("rollout/success_rate")
        return events[-1].step if events else 0
    except Exception:
        return 0


def _best_tb_run(algo_path: str, tb_prefix: str) -> Path | None:
    base: Path = LOGS_DIR / algo_path
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir()
                  if p.is_dir() and p.name.startswith(f"{tb_prefix}_")]
    if not candidates:
        return None
    best = max(candidates, key=_tb_run_steps)
    return best if _tb_run_steps(best) > 0 else None


def load_eval(algo_path: str) -> dict[str, np.ndarray] | None:
    fpath: Path = LOGS_DIR / algo_path / "evaluations.npz"
    if not fpath.exists():
        return None
    raw        = dict(np.load(fpath))
    ts         = raw["timesteps"]
    results    = raw["results"]
    ep_lens    = raw["ep_lengths"]
    successes  = raw["successes"].astype(bool)
    n_eval     = successes.shape[1]

    success_rate = successes.mean(axis=1)
    success_std  = successes.std(axis=1)

    precision     = np.full(len(ts), np.nan)
    precision_std = np.full(len(ts), np.nan)
    for i, mask in enumerate(successes):
        if mask.any():
            vals = results[i][mask]
            precision[i]     = vals.mean()
            precision_std[i] = vals.std()

    valid = ~np.isnan(precision)
    if valid.any():
        prec_max = float(np.nanmax(precision))
        prec_min = float(np.nanmin(precision))
        if prec_max > prec_min:
            precision[valid]     = (precision[valid] - prec_min) / (prec_max - prec_min)
            precision_std[valid] = precision_std[valid] / (prec_max - prec_min)
        else:
            precision[valid]     = 1.0
            precision_std[valid] = 0.0

    steps_solve     = np.full(len(ts), np.nan)
    steps_solve_std = np.full(len(ts), np.nan)
    for i, mask in enumerate(successes):
        if mask.any():
            vals = ep_lens[i][mask].astype(float)
            steps_solve[i]     = vals.mean()
            steps_solve_std[i] = vals.std()

    return {
        "timesteps":      ts,
        "success_rate":   success_rate,
        "success_std":    success_std,
        "precision":      precision,
        "precision_std":  precision_std,
        "steps_to_solve": steps_solve,
        "steps_solve_std":steps_solve_std,
        "n_eval":         np.array(n_eval),
    }


def load_tb_scalars(run_path: Path, tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(run_path), size_guidance={"scalars": 0})
        ea.Reload()
        available = ea.Tags().get("scalars", [])
        result = {}
        for tag in tags:
            if tag in available:
                events     = ea.Scalars(tag)
                steps      = np.array([e.step  for e in events])
                vals       = np.array([e.value for e in events])
                result[tag] = (steps, vals)
        return result
    except Exception:
        return {}


# ── Utilitaires graphiques ─────────────────────────────────────────────────────

def _millions(x: float, _pos: int) -> str:
    if x >= 1_000_000:
        return f"{x/1e6:.1f}M"
    if x >= 1_000:
        return f"{int(x/1_000)}k"
    return str(int(x))


def _smooth(xs: np.ndarray, ys: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(ys) <= window:
        return xs, ys
    kernel   = np.ones(window) / window
    ys_smooth = np.convolve(ys, kernel, mode="valid")
    return xs[window - 1:], ys_smooth


def _plot_band(ax, xs, mean, std, color, linestyle="-", label="",
               alpha_fill=0.15, linewidth=2.0, smooth_window=0) -> None:
    if smooth_window > 1:
        xs_s, mean_s = _smooth(xs, mean, smooth_window)
        _,    std_s  = _smooth(xs, std,  smooth_window)
    else:
        xs_s, mean_s, std_s = xs, mean, std
    ax.plot(xs_s, mean_s, color=color, linestyle=linestyle,
            linewidth=linewidth, label=label)
    ax.fill_between(xs_s, mean_s - std_s, mean_s + std_s,
                    alpha=alpha_fill, color=color)


# ── ① Nouvelles fonctions utilitaires pour axes dynamiques ────────────────────  ← DYNAMIC

def _data_ylim(
    arrays: list[np.ndarray],
    margin: float = 0.08,
    clamp_bottom: float | None = None,
    clamp_top:    float | None = None,
) -> tuple[float, float]:
    filtered = [a[~np.isnan(a)].ravel() for a in arrays if a.size]  # ← DYNAMIC
    filtered = [a for a in filtered if len(a)]                       # ← filtre les tableaux vides après NaN-drop
    if not filtered:                                                  # ← garde-fou : liste vide
        lo = clamp_bottom if clamp_bottom is not None else 0.0
        hi = clamp_top    if clamp_top    is not None else 1.0
        return (lo, hi)

    all_vals = np.concatenate(filtered)
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    span = (vmax - vmin) or 1.0
    lo   = vmin - margin * span
    hi   = vmax + margin * span

    if clamp_bottom is not None:
        lo = max(lo, clamp_bottom)
    if clamp_top is not None:
        hi = min(hi, clamp_top)
    return (lo, hi)

def _global_xlim(
    eval_data: dict[str, dict[str, np.ndarray]],
    tb_data:   dict[str, dict[str, Any]],
    margin:    float = 0.02,
) -> tuple[float, float] | None:
    """Retourne (xmin, xmax) couvrant l'ensemble des timesteps de tous les algos.

    Prend en compte les timesteps d'évaluation ET les steps TensorBoard,
    de sorte que toutes les courbes s'inscrivent dans la même fenêtre X.
    """                                                                   # ← DYNAMIC
    all_ts: list[float] = []
    for d in eval_data.values():
        all_ts.extend(d["timesteps"].tolist())
    for tb in tb_data.values():
        for xs, _ in tb.values():
            all_ts.extend(xs.tolist())
    if not all_ts:
        return None
    xmax = max(all_ts)
    return (0.0, xmax * (1.0 + margin))


# ── ② _style_ax : ylim dynamique + ticks adaptatifs ──────────────────────────  ← DYNAMIC

def _style_ax(
    ax: plt.Axes,
    title: str,
    ylabel: str,
    ylim:      tuple[float | None, float | None] = (None, None),
    percent_y: bool = False,
) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Timesteps d'entraînement", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_millions))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=False))  # ← DYNAMIC

    if percent_y:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))               # ← DYNAMIC

    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    if ylim[0] is not None or ylim[1] is not None:                         # ← DYNAMIC
        ax.set_ylim(ylim)
    else:
        ax.autoscale_view()
        ax.margins(y=0.08)


# ── ③ Tracé des 4 métriques avec ylim calculé depuis les données ──────────────

def _plot_success_rate(ax, eval_data, tb_data) -> None:
    # collecte toutes les valeurs pour calculer ylim dynamiquement
    all_sr: list[np.ndarray] = []                                          # ← DYNAMIC

    for name, data in eval_data.items():
        cfg = ALGOS[name]
        _plot_band(ax, data["timesteps"], data["success_rate"],
                   data["success_std"], cfg["color"],
                   linestyle="-", label=f"{name} (test)", smooth_window=5)
        all_sr.append(data["success_rate"])                                # ← DYNAMIC

    for name, tb in tb_data.items():
        if "rollout/success_rate" in tb:
            xs, ys = tb["rollout/success_rate"]
            xs_s, ys_s = _smooth(xs, ys, 30)
            cfg = ALGOS[name]
            ax.plot(xs_s, ys_s, color=cfg["color"], linestyle="--",
                    linewidth=1.4, alpha=0.7, label=f"{name} (train)")
            all_sr.append(ys_s)                                            # ← DYNAMIC

    ylim = _data_ylim(all_sr, margin=0.05,                                 # ← DYNAMIC
                      clamp_bottom=0.0, clamp_top=1.05)
    _style_ax(ax, "Taux de succès", "Success Rate",
              ylim=ylim, percent_y=True)
    ax.legend(fontsize=7, ncols=2)


def _plot_precision(ax, eval_data) -> None:
    all_prec: list[np.ndarray] = []                                        # ← DYNAMIC

    for name, data in eval_data.items():
        cfg   = ALGOS[name]
        valid = ~np.isnan(data["precision"])
        if not valid.any():
            continue
        ax.plot(data["timesteps"][valid], data["precision"][valid],
                color=cfg["color"], linestyle="-", linewidth=2,
                label=name, marker=cfg["marker"],
                markevery=max(1, valid.sum() // 15), markersize=4)
        ax.fill_between(
            data["timesteps"][valid],
            np.clip(data["precision"][valid] - data["precision_std"][valid], 0, 1),
            np.clip(data["precision"][valid] + data["precision_std"][valid], 0, 1),
            alpha=0.12, color=cfg["color"])
        all_prec.append(data["precision"][valid])                          # ← DYNAMIC

    ylim = _data_ylim(all_prec, margin=0.05,                               # ← DYNAMIC
                      clamp_bottom=0.0, clamp_top=1.05)
    _style_ax(ax,
              "Précision des épisodes réussis\n(normalisée par algo)",
              "Précision normalisée [0 – 1]", ylim=ylim)
    ax.legend(fontsize=8)
    ax.text(0.01, 0.02, "1 = meilleure récompense observée pour cet algo",
            transform=ax.transAxes, fontsize=7, color="grey", style="italic")


def _plot_steps_to_solve(ax, eval_data) -> None:
    all_steps: list[np.ndarray] = []                                       # ← DYNAMIC

    for name, data in eval_data.items():
        cfg   = ALGOS[name]
        valid = ~np.isnan(data["steps_to_solve"])
        if not valid.any():
            continue
        ax.errorbar(
            data["timesteps"][valid], data["steps_to_solve"][valid],
            yerr=data["steps_solve_std"][valid],
            color=cfg["color"], linestyle="-", linewidth=2, label=name,
            marker=cfg["marker"], markevery=max(1, valid.sum() // 15),
            markersize=4, elinewidth=0.8, capsize=2,
            errorevery=max(1, valid.sum() // 8))
        # inclut mean ± std dans le calcul des limites
        all_steps.append(data["steps_to_solve"][valid])                    # ← DYNAMIC
        all_steps.append(data["steps_to_solve"][valid]
                         + data["steps_solve_std"][valid])

    ylim = _data_ylim(all_steps, margin=0.08, clamp_bottom=0.0)            # ← DYNAMIC
    _style_ax(ax,
              "Steps nécessaires pour réussir\n(épisodes réussis uniquement)",
              "Steps moyen à la réussite", ylim=ylim)
    ax.legend(fontsize=8)


def _plot_generalizability(ax, eval_data, tb_data) -> None:
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

    all_gaps: list[np.ndarray] = []                                        # ← DYNAMIC

    for name, data in eval_data.items():
        tb = tb_data.get(name, {})
        if "rollout/success_rate" not in tb:
            continue
        cfg                = ALGOS[name]
        ts_eval            = data["timesteps"]
        sr_eval            = data["success_rate"]
        ts_train, sr_train = tb["rollout/success_rate"]
        xs_s, ys_s         = _smooth(ts_train, sr_train, 30)
        sr_train_interp    = np.interp(ts_eval, xs_s, ys_s)
        gap                = sr_eval - sr_train_interp

        ax.plot(ts_eval, gap, color=cfg["color"], linestyle="-",
                linewidth=2, label=name)
        ax.fill_between(ts_eval, gap, 0, alpha=0.08, color=cfg["color"])
        all_gaps.append(gap)                                               # ← DYNAMIC

    # ylim symétrique autour de 0, amplitude pilotée par les données
    if all_gaps:                                                            # ← DYNAMIC
        amp = float(np.nanmax(np.abs(np.concatenate(all_gaps)))) * 1.25
        ylim = (-amp, amp)
    else:
        ylim = (-0.1, 0.1)

    _style_ax(ax,
              "Généralisabilité\n(Succès test − Succès train)",
              "Δ Success Rate (test − train)",
              ylim=ylim)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=8)
    ax.text(0.01, 0.97,
            "> 0 : meilleur en test  |  < 0 : sur-apprentissage exploration",
            transform=ax.transAxes, fontsize=7, color="grey",
            style="italic", va="top")


# ── Légende commune (inchangée) ────────────────────────────────────────────────

def _add_train_test_legend(fig: plt.Figure) -> None:
    legend_handles = [
        mlines.Line2D([], [], color="black", linestyle="-",  linewidth=2,
                      label="Test (évaluation déterministe)"),
        mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.4,
                      alpha=0.7, label="Train (rollout stochastique)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncols=2,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    eval_data: dict[str, dict[str, np.ndarray]] = {}
    tb_data:   dict[str, dict[str, Any]]        = {}

    for name, cfg in ALGOS.items():
        d = load_eval(cfg["path"])
        if d is None:
            print(f"[{name}] evaluations.npz absent — algo ignoré")
            continue
        eval_data[name] = d

        best_run = _best_tb_run(cfg["path"], cfg["tb_prefix"])
        if best_run is not None:
            tb_data[name] = load_tb_scalars(
                best_run, ["rollout/success_rate", "eval/success_rate"])
            print(f"[{name}] TB run : {best_run.name}")
        else:
            print(f"[{name}] aucun run TensorBoard trouvé")

    if not eval_data:
        print("Aucune donnée disponible.")
        return

    # ── calcul du xlim global ─────────────────────────────────────────────────  ← DYNAMIC
    xlim_global = _global_xlim(eval_data, tb_data)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Benchmark algorithmique — Robot 3-DDL (MuJoCo)\n"
        "Comparaison PPO / HER-SAC / TD3",
        fontsize=14, fontweight="bold", y=1.01)

    _plot_success_rate(axes[0, 0], eval_data, tb_data)
    _plot_precision(   axes[0, 1], eval_data)
    _plot_steps_to_solve(axes[1, 0], eval_data)
    _plot_generalizability(axes[1, 1], eval_data, tb_data)

    # ── applique le même xlim à tous les axes ─────────────────────────────────  ← DYNAMIC
    if xlim_global is not None:
        for ax in axes.flat:
            ax.set_xlim(xlim_global)

    _add_train_test_legend(fig)
    plt.tight_layout(rect=[0, 0.02, 1, 1])

    out_path = OUT_DIR / "benchmark_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure sauvegardée : {out_path}")
    plt.show()


if __name__ == "__main__":
    main()