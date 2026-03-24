"""Benchmark comparatif PPO / HER-SAC / TD3 sur les tâches robotiques 3-DDL.

Métriques calculées pour chaque algorithme :
- Taux de succès (train et test) — is_success
- Précision de la solution — récompense moyenne des épisodes réussis,
  normalisée par sa valeur maximale observée
- Timesteps moyens pour résoudre la tâche — ep_length moyen sur les épisodes
  réussis uniquement
- Généralisabilité — écart entre succès en évaluation (déterministe, test)
  et succès en entraînement (stochastique, avec bruit d'exploration)

Sources de données :
- ``evaluations.npz``  : données de test, produites par EvalCallback de SB3
                         (politique déterministe, sans bruit d'exploration)
- TensorBoard events   : données d'entraînement (rollout/success_rate),
                         run le plus complet par algorithme

TD3 est inclus automatiquement dès que son dossier ``logs/td3/`` est présent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np

LOGS_DIR: Path = Path(__file__).parent / "logs"
OUT_DIR: Path = Path(__file__).parent

#algorithmes ciblés par ce benchmark ; TD3 est inclus si son dossier existe
ALGOS: dict[str, dict[str, Any]] = {
    "HER-SAC": {
        "path": "her_sac",
        "tb_prefix": "SAC",
        "color": "#2196F3",
        "marker": "o",
    },
    "PPO": {
        "path": "ppo",
        "tb_prefix": "PPO",
        "color": "#F44336",
        "marker": "s",
    },
    "TD3": {
        "path": "td3",
        "tb_prefix": "TD3",
        "color": "#4CAF50",
        "marker": "^",
    },
}


# ── Chargement des données ────────────────────────────────────────────────────

def _tb_run_steps(run_path: Path) -> int:
    """Retourne le nombre de pas dans le dernier event du run, ou 0 si vide.
    Parameters:
        run_path (Path): dossier d'un run TensorBoard
    Returns:
        int: dernier step présent dans rollout/success_rate, ou 0.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
        ea: EventAccumulator = EventAccumulator(
            str(run_path), size_guidance={"scalars": 0}
        )
        ea.Reload()
        events = ea.Scalars("rollout/success_rate")
        return events[-1].step if events else 0
    except Exception:
        return 0


def _best_tb_run(algo_path: str, tb_prefix: str) -> Path | None:
    """Sélectionne le run TensorBoard le plus complet pour un algorithme.

    Cherche dans ``LOGS_DIR/algo_path/`` les sous-dossiers nommés
    ``{tb_prefix}_N``, et retourne celui dont le dernier step est le plus grand
    (= run le plus avancé en entraînement).
    Parameters:
        algo_path (str): sous-dossier de l'algorithme dans logs/
        tb_prefix (str): préfixe des runs, ex. ``"SAC"`` ou ``"PPO"``
    Returns:
        Path | None: chemin du run le plus complet, ou None si aucun.
    """
    base: Path = LOGS_DIR / algo_path
    if not base.exists():
        return None
    candidates: list[Path] = [
        p for p in base.iterdir()
        if p.is_dir() and p.name.startswith(f"{tb_prefix}_")
    ]
    if not candidates:
        return None
    best: Path = max(candidates, key=_tb_run_steps)
    return best if _tb_run_steps(best) > 0 else None


def load_eval(algo_path: str) -> dict[str, np.ndarray] | None:
    """Charge et enrichit le fichier evaluations.npz d'un algorithme.

    Calcule les métriques dérivées suivantes :
    - ``success_rate`` : taux de succès par checkpoint (moyenne sur les N_EVAL épisodes)
    - ``success_std``  : écart-type du taux de succès
    - ``precision``    : récompense moyenne des épisodes réussis (proxy de qualité)
    - ``steps_to_solve``: longueur d'épisode moyenne quand is_success=True
    - ``steps_to_solve_std``: écart-type de la longueur d'épisode sur les épisodes réussis

    ``precision`` et ``steps_to_solve`` valent NaN aux checkpoints sans succès.
    Parameters:
        algo_path (str): sous-dossier dans logs/
    Returns:
        dict[str, np.ndarray]: métriques par checkpoint, ou None si absent.
    """
    fpath: Path = LOGS_DIR / algo_path / "evaluations.npz"
    if not fpath.exists():
        return None

    raw: dict[str, np.ndarray] = dict(np.load(fpath))
    ts: np.ndarray       = raw["timesteps"]               #(T,)
    results: np.ndarray  = raw["results"]                  #(T, N_eval)
    ep_lens: np.ndarray  = raw["ep_lengths"]               #(T, N_eval)
    successes: np.ndarray = raw["successes"].astype(bool)  #(T, N_eval)

    n_eval: np.ndarray = successes.shape[1]

    #taux de succès par checkpoint
    success_rate: np.ndarray = successes.mean(axis=1)
    success_std: np.ndarray  = successes.std(axis=1)

    #récompense moyenne des épisodes réussis (NaN si aucun succès)
    precision: np.ndarray     = np.full(len(ts), np.nan)
    precision_std: np.ndarray = np.full(len(ts), np.nan)
    for i, mask in enumerate(successes):
        if mask.any():
            vals: np.ndarray = results[i][mask]
            precision[i]     = vals.mean()
            precision_std[i] = vals.std()

    #normalisation [0, 1] relative au maximum observé (comparaison intra-algo)
    valid: np.ndarray = ~np.isnan(precision)
    if valid.any():
        prec_max: float  = float(np.nanmax(precision))
        prec_min: float  = float(np.nanmin(precision))
        if prec_max > prec_min:
            precision[valid] = (
                (precision[valid] - prec_min) / (prec_max - prec_min)
            )
            precision_std[valid] = precision_std[valid] / (prec_max - prec_min)
        else:
            precision[valid] = 1.0
            precision_std[valid] = 0.0

    #longueur d'épisode moyenne quand is_success=True
    steps_solve: np.ndarray     = np.full(len(ts), np.nan)
    steps_solve_std: np.ndarray = np.full(len(ts), np.nan)
    for i, mask in enumerate(successes):
        if mask.any():
            vals = ep_lens[i][mask].astype(float)
            steps_solve[i]     = vals.mean()
            steps_solve_std[i] = vals.std()

    return {
        "timesteps":        ts,
        "success_rate":     success_rate,
        "success_std":      success_std,
        "precision":        precision,
        "precision_std":    precision_std,
        "steps_to_solve":   steps_solve,
        "steps_solve_std":  steps_solve_std,
        "n_eval":           np.array(n_eval),
    }


def load_tb_scalars(
    run_path: Path,
    tags: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Charge des scalaires TensorBoard depuis un run.

    Parameters:
        run_path (Path): dossier du run
        tags (list[str]): tags scalaires à extraire
    Returns:
        dict[str, (steps, values)]: tableaux numpy par tag.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
        ea: EventAccumulator = EventAccumulator(
            str(run_path), size_guidance={"scalars": 0}
        )
        ea.Reload()
        available: list[str] = ea.Tags().get("scalars", [])
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                steps: np.ndarray = np.array([e.step for e in events])
                vals: np.ndarray  = np.array([e.value for e in events])
                result[tag] = (steps, vals)
        return result
    except Exception:
        return {}


# ── Utilitaires graphiques ────────────────────────────────────────────────────

def _millions(x: float, _pos: int) -> str:
    """Formatte l'axe X des timesteps en k / M."""
    if x >= 1_000_000:
        return f"{x/1e6:.1f}M"
    if x >= 1_000:
        return f"{int(x/1_000)}k"
    return str(int(x))


def _smooth(
    xs: np.ndarray,
    ys: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne mobile d'ordre ``window`` sur ``ys``.

    Parameters:
        xs (np.ndarray): vecteur des abscisses
        ys (np.ndarray): vecteur des valeurs
        window (int): taille de la fenêtre de lissage
    Returns:
        (xs_trimmed, ys_smoothed): abscisses et valeurs lissées.
    """
    if len(ys) <= window:
        return xs, ys
    kernel: np.ndarray   = np.ones(window) / window
    ys_smooth: np.ndarray = np.convolve(ys, kernel, mode="valid")
    return xs[window - 1:], ys_smooth


def _plot_band(
    ax: plt.Axes,
    xs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    linestyle: str = "-",
    label: str = "",
    alpha_fill: float = 0.15,
    linewidth: float = 2.0,
    smooth_window: int = 0,
) -> None:
    """Trace une courbe avec bande ±1 std, optionnellement lissée.

    Parameters:
        ax (plt.Axes): axes cible
        xs (np.ndarray): abscisses
        mean (np.ndarray): valeurs centrales
        std (np.ndarray): écart-type par point
        color (str): couleur hex
        linestyle (str): style de trait
        label (str): entrée de légende
        alpha_fill (float): opacité de la bande
        linewidth (float): épaisseur du trait
        smooth_window (int): fenêtre de lissage (0 = pas de lissage)
    """
    if smooth_window > 1:
        xs_s, mean_s = _smooth(xs, mean, smooth_window)
        _, std_s = _smooth(xs, std, smooth_window)
    else:
        xs_s, mean_s, std_s = xs, mean, std
    ax.plot(xs_s, mean_s, color=color, linestyle=linestyle,
            linewidth=linewidth, label=label)
    ax.fill_between(
        xs_s, mean_s - std_s, mean_s + std_s,
        alpha=alpha_fill, color=color,
    )


def _style_ax(
    ax: plt.Axes,
    title: str,
    ylabel: str,
    ylim: tuple[float | None, float | None] = (None, None),
    percent_y: bool = False,
) -> None:
    """Applique le style commun aux axes du benchmark.

    Parameters:
        ax (plt.Axes): axes à styler
        title (str): titre du sous-graphique
        ylabel (str): label de l'axe Y
        ylim (tuple): (ymin, ymax), None = automatique
        percent_y (bool): formatte l'axe Y en pourcentage
    """
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Timesteps d'entraînement", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_millions))
    if percent_y:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim[0] is not None or ylim[1] is not None:
        ax.set_ylim(ylim)


# ── Tracé des 4 métriques ─────────────────────────────────────────────────────

def _plot_success_rate(
    ax: plt.Axes,
    eval_data: dict[str, dict[str, np.ndarray]],
    tb_data: dict[str, dict[str, Any]],
) -> None:
    """Taux de succès en test (eval) et entraînement (rollout) par algo.

    Lignes pleines = évaluation déterministe, lignes tiretées = rollout.
    Parameters:
        ax (plt.Axes): axes cible
        eval_data (dict): données des evaluations.npz par algo
        tb_data (dict): données TensorBoard (rollout/success_rate) par algo
    """
    for name, data in eval_data.items():
        cfg: dict[str, Any] = ALGOS[name]
        _plot_band(
            ax,
            data["timesteps"],
            data["success_rate"],
            data["success_std"],
            cfg["color"],
            linestyle="-",
            label=f"{name} (test)",
            smooth_window=5,
        )

    for name, tb in tb_data.items():
        if "rollout/success_rate" in tb:
            xs, ys = tb["rollout/success_rate"]
            xs_s, ys_s = _smooth(xs, ys, 30)
            cfg = ALGOS[name]
            ax.plot(
                xs_s, ys_s,
                color=cfg["color"],
                linestyle="--",
                linewidth=1.4,
                alpha=0.7,
                label=f"{name} (train)",
            )

    _style_ax(
        ax,
        "Taux de succès",
        "Success Rate",
        ylim=(0, 1.05),
        percent_y=True,
    )
    ax.legend(fontsize=7, ncols=2)


def _plot_precision(
    ax: plt.Axes,
    eval_data: dict[str, dict[str, np.ndarray]],
) -> None:
    """Précision normalisée des épisodes réussis.

    Définie comme la récompense moyenne des épisodes réussis, normalisée
    en [0, 1] par rapport à la plage observée pour chaque algorithme.
    Un score de 1 correspond au meilleur épisode réussi observé.

    N.B. : les environnements utilisés par chaque algo ont des récompenses
    différentes — cette métrique n'est PAS comparable entre algos, mais
    illustre la progression de la qualité au sein de chaque entraînement.
    Parameters:
        ax (plt.Axes): axes cible
        eval_data (dict): données des evaluations.npz par algo
    """
    for name, data in eval_data.items():
        cfg: dict[str, Any] = ALGOS[name]
        valid: np.ndarray = ~np.isnan(data["precision"])
        if not valid.any():
            continue
        ax.plot(
            data["timesteps"][valid],
            data["precision"][valid],
            color=cfg["color"],
            linestyle="-",
            linewidth=2,
            label=name,
            marker=cfg["marker"],
            markevery=max(1, valid.sum() // 15),
            markersize=4,
        )
        ax.fill_between(
            data["timesteps"][valid],
            np.clip(data["precision"][valid] - data["precision_std"][valid], 0, 1),
            np.clip(data["precision"][valid] + data["precision_std"][valid], 0, 1),
            alpha=0.12,
            color=cfg["color"],
        )

    _style_ax(
        ax,
        "Précision des épisodes réussis\n(normalisée par algo)",
        "Précision normalisée [0 – 1]",
        ylim=(0, 1.05),
    )
    ax.legend(fontsize=8)
    ax.text(
        0.01, 0.02,
        "1 = meilleure récompense observée pour cet algo",
        transform=ax.transAxes,
        fontsize=7,
        color="grey",
        style="italic",
    )


def _plot_steps_to_solve(
    ax: plt.Axes,
    eval_data: dict[str, dict[str, np.ndarray]],
) -> None:
    """Nombre de steps moyens pour résoudre la tâche.

    Calculé uniquement sur les épisodes ayant abouti à un succès.
    NaN = aucun succès à ce checkpoint.
    Parameters:
        ax (plt.Axes): axes cible
        eval_data (dict): données des evaluations.npz par algo
    """
    for name, data in eval_data.items():
        cfg: dict[str, Any] = ALGOS[name]
        valid: np.ndarray = ~np.isnan(data["steps_to_solve"])
        if not valid.any():
            continue
        ax.errorbar(
            data["timesteps"][valid],
            data["steps_to_solve"][valid],
            yerr=data["steps_solve_std"][valid],
            color=cfg["color"],
            linestyle="-",
            linewidth=2,
            label=name,
            marker=cfg["marker"],
            markevery=max(1, valid.sum() // 15),
            markersize=4,
            elinewidth=0.8,
            capsize=2,
            errorevery=max(1, valid.sum() // 8),
        )

    _style_ax(
        ax,
        "Steps nécessaires pour réussir\n(épisodes réussis uniquement)",
        "Steps moyen à la réussite",
    )
    ax.legend(fontsize=8)


def _plot_generalizability(
    ax: plt.Axes,
    eval_data: dict[str, dict[str, np.ndarray]],
    tb_data: dict[str, dict[str, Any]],
) -> None:
    """Généralisabilité : écart entre succès entraînement et test.

    Un écart positif indique que la politique évalue mieux en mode
    déterministe qu'en mode stochastique (bon signe).
    Un écart négatif indique un surapprentissage au bruit d'exploration.
    Un écart proche de 0 indique une bonne consistance entraînement/test.

    L'écart est interpolé au plus proche voisin pour aligner les pas
    des deux sources (TB tous les ~400 steps, eval tous les 5k–10k steps).
    Parameters:
        ax (plt.Axes): axes cible
        eval_data (dict): données eval (test, déterministe)
        tb_data (dict): données TB (train, stochastique)
    """
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

    for name, data in eval_data.items():
        tb: dict[str, Any] = tb_data.get(name, {})
        if "rollout/success_rate" not in tb:
            continue
        cfg: dict[str, Any] = ALGOS[name]

        ts_eval: np.ndarray  = data["timesteps"]
        sr_eval: np.ndarray  = data["success_rate"]

        ts_train, sr_train = tb["rollout/success_rate"]
        xs_s, ys_s = _smooth(ts_train, sr_train, 30)

        #interpolation du rollout sur les timesteps de l'éval
        sr_train_interp: np.ndarray = np.interp(ts_eval, xs_s, ys_s)
        gap: np.ndarray = sr_eval - sr_train_interp

        ax.plot(
            ts_eval, gap,
            color=cfg["color"],
            linestyle="-",
            linewidth=2,
            label=name,
        )
        ax.fill_between(ts_eval, gap, 0, alpha=0.08, color=cfg["color"])

    _style_ax(
        ax,
        "Généralisabilité\n(Succès test − Succès train)",
        "Δ Success Rate (test − train)",
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=8)
    ax.text(
        0.01, 0.97,
        "> 0 : meilleur en test  |  < 0 : sur-apprentissage exploration",
        transform=ax.transAxes, fontsize=7, color="grey",
        style="italic", va="top",
    )


# ── Légende commune entraînement / test ──────────────────────────────────────

def _add_train_test_legend(fig: plt.Figure) -> None:
    """Ajoute une légende globale distinguant courbes entraînement et test.
    Parameters:
        fig (plt.Figure): figure cible
    """
    legend_handles: list[mlines.Line2D] = [
        mlines.Line2D([], [], color="black", linestyle="-",  linewidth=2,
                      label="Test (évaluation déterministe)"),
        mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.4,
                      alpha=0.7, label="Train (rollout stochastique)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncols=2,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )


# ── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    """Charge les données, filtre les algos disponibles et génère la figure."""
    eval_data: dict[str, dict[str, np.ndarray]] = {}
    tb_data:   dict[str, dict[str, Any]]        = {}

    for name, cfg in ALGOS.items():
        d: dict[str, np.ndarray] | None = load_eval(cfg["path"])
        if d is None:
            print(f"[{name}] evaluations.npz absent — algo ignoré")
            continue
        eval_data[name] = d

        best_run: Path | None = _best_tb_run(cfg["path"], cfg["tb_prefix"])
        if best_run is not None:
            tb_data[name] = load_tb_scalars(
                best_run,
                ["rollout/success_rate", "eval/success_rate"],
            )
            print(f"[{name}] TB run : {best_run.name}")
        else:
            print(f"[{name}] aucun run TensorBoard trouvé")

    if not eval_data:
        print("Aucune donnée disponible.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Benchmark algorithmique — Robot 3-DDL (MuJoCo)\n"
        "Comparaison PPO / HER-SAC / TD3",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    _plot_success_rate(axes[0, 0], eval_data, tb_data)
    _plot_precision(axes[0, 1], eval_data)
    _plot_steps_to_solve(axes[1, 0], eval_data)
    _plot_generalizability(axes[1, 1], eval_data, tb_data)

    _add_train_test_legend(fig)
    plt.tight_layout(rect=[0, 0.02, 1, 1])

    out_path: Path = OUT_DIR / "benchmark_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure sauvegardée : {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
