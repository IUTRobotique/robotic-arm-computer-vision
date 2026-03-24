"""YoloBliss — Page unique : algo -> env -> mode -> resultats."""
import json
import os
import subprocess
import sys
import tempfile

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from utils.paths import ALGO_INFO, ENV_NAMES, RUN_EPISODE_SCRIPT, model_path, log_path  # noqa
from utils.style import inject_css  # noqa

st.set_page_config(page_title="YoloBliss", layout="wide", initial_sidebar_state="collapsed")
inject_css()

for key, value in {
    "algo": None,
    "env": "reaching",
    "mode": "Simulation",
    "result": None,
    "running": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def _pick_algo(name: str) -> None:
    st.session_state.algo = name
    st.session_state.env = ALGO_INFO[name]["env"]
    st.session_state.result = None


def _read_tb_logs(algo_key: str):
    """Lit les logs tensorboard SB3 et retourne un DataFrame {step, reward}."""
    import glob
    import pandas as pd
    ldir = log_path(algo_key)
    if not os.path.isdir(ldir):
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    runs = sorted(glob.glob(os.path.join(ldir, "*")))
    if not runs:
        return None
    frames = []
    for run in runs:
        try:
            ea = EventAccumulator(run, size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            tag  = next((t for t in ("rollout/ep_rew_mean", "train/reward") if t in tags), None)
            if tag is None:
                continue
            events = ea.Scalars(tag)
            run_name = os.path.basename(run)
            for e in events:
                frames.append({"run": run_name, "step": e.step, "reward": e.value})
        except Exception:
            continue
    if not frames:
        return None
    return pd.DataFrame(frames)


st.title("YoloBliss")
st.caption("Choisir un algorithme, un environnement, un mode, puis voir les resultats.")

# ══════════════════════════════════════════════════════════════════════════════
#  MISE EN PAGE  ──  Controles (gauche)  |  Resultats (droite)
# ══════════════════════════════════════════════════════════════════════════════
col_ctrl, col_res = st.columns([2, 3], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
#  CONTROLES
# ─────────────────────────────────────────────────────────────────────────────
with col_ctrl:
    st.subheader("Configuration")

    algo_names = list(ALGO_INFO.keys())
    default_algo_index = algo_names.index(st.session_state.algo) if st.session_state.algo in algo_names else 0
    algo_choice = st.selectbox("Algorithme", algo_names, index=default_algo_index)
    if algo_choice != st.session_state.algo:
        _pick_algo(algo_choice)

    st.write(ALGO_INFO[st.session_state.algo]["description"])

    current_model_path = model_path(st.session_state.algo)
    if os.path.exists(current_model_path):
        st.success("Modele disponible")
    else:
        st.warning("Modele introuvable : la simulation utilisera une politique aleatoire.")

    env_keys = list(ENV_NAMES.keys())
    env_labels = [ENV_NAMES[key] for key in env_keys]
    current_env_index = env_keys.index(st.session_state.env) if st.session_state.env in env_keys else 0
    env_choice = st.radio("Environnement", env_labels, index=current_env_index)
    st.session_state.env = env_keys[env_labels.index(env_choice)]

    mode_options = ["Simulation", "Robot Reel"]
    mode_choice = st.radio("Mode", mode_options, index=mode_options.index(st.session_state.mode), horizontal=True)
    st.session_state.mode = mode_choice

    if st.button("Lancer", type="primary", use_container_width=True):
        st.session_state.running = True
        st.session_state.result = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTATS
# ─────────────────────────────────────────────────────────────────────────────
with col_res:
    st.subheader("Resultats")

    # ── Execution en cours ────────────────────────────────────────────────────
    if st.session_state.running:
        algo_key  = st.session_state.algo
        env_key   = st.session_state.env
        mode_key  = st.session_state.mode
        _mpath    = model_path(algo_key)

        if mode_key == "Simulation":
            out_dir = tempfile.mkdtemp(prefix="yb_")
            with st.spinner(f"Simulation : {algo_key}  /  {ENV_NAMES.get(env_key, env_key)} ..."):
                try:
                    proc = subprocess.run(
                        [sys.executable, RUN_EPISODE_SCRIPT,
                         env_key, _mpath if os.path.exists(_mpath) else "none",
                         out_dir, "300"],
                        capture_output=True, text=True, timeout=180,
                    )
                    _mf = os.path.join(out_dir, "metrics.json")
                    if os.path.exists(_mf):
                        with open(_mf) as _f:
                            res = json.load(_f)
                        res["_algo"] = algo_key
                        res["_env"]  = env_key
                        res["_mode"] = "sim"
                    else:
                        res = {"error": (proc.stderr or proc.stdout or "Erreur inconnue")[:800],
                               "_mode": "sim"}
                except subprocess.TimeoutExpired:
                    res = {"error": "Timeout (> 3 min).", "_mode": "sim"}
                except Exception as e:
                    res = {"error": str(e), "_mode": "sim"}
        else:
            res = {"_mode": "reel", "_algo": algo_key, "_env": env_key, "_mpath": _mpath}

        st.session_state.result  = res
        st.session_state.running = False
        st.rerun()

    # ── Affichage ─────────────────────────────────────────────────────────────
    result = st.session_state.result

    if result is None:
        # Etat initial : afficher courbes de logs si algo deja choisi
        selected_algo = st.session_state.algo
        if selected_algo:
            df_logs = _read_tb_logs(selected_algo)
            if df_logs is not None:
                import plotly.graph_objects as go
                fig = go.Figure()
                for run_name in df_logs["run"].unique():
                    sub = df_logs[df_logs["run"] == run_name]
                    fig.add_trace(go.Scatter(
                        x=sub["step"], y=sub["reward"],
                        mode="lines", name=run_name,
                        line=dict(width=1.5),
                    ))
                fig.update_layout(
                    title=f"Courbe d'entrainement — {selected_algo}",
                    xaxis_title="Steps",
                    yaxis_title="Recompense moyenne",
                    template="plotly_dark",
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#0d1117",
                    height=420,
                    legend=dict(font=dict(size=10)),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Aucun log d'entrainement disponible pour cet algorithme.")
                st.write("Appuyez sur Lancer pour demarrer une simulation.")
        else:
            st.info("Choisissez un algorithme, un environnement, un mode, puis appuyez sur Lancer.")

    elif "error" in result:
        st.error(result["error"])

    elif result.get("_mode") == "sim":
        # ── Resultat simulation ───────────────────────────────────────────────
        vpath = result.get("video_path", "")
        if vpath and os.path.exists(vpath):
            st.video(vpath)

        _reward  = result.get("total_reward")
        _steps   = result.get("n_steps")
        _success = result.get("is_success")
        _sc_str  = "Oui" if _success else "Non" if _success is not None else "—"
        _env_lbl = ENV_NAMES.get(result.get("_env", ""), result.get("_env", "—"))
        _mdl_lbl = "Aleatoire" if result.get("model") == "random" else os.path.basename(result.get("model", "—"))

        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Recompense totale", f"{_reward:.3f}" if isinstance(_reward, float) else "—")
        metric_b.metric("Etapes", _steps if _steps is not None else "—")
        metric_c.metric("Succes", _sc_str)

        st.write(f"Algorithme : {result.get('_algo', '—')}")
        st.write(f"Environnement : {_env_lbl}")
        st.write(f"Modele : {_mdl_lbl}")

        # Afficher aussi la courbe d'entrainement en dessous
        df_logs = _read_tb_logs(result.get("_algo", ""))
        if df_logs is not None:
            import plotly.graph_objects as go
            fig2 = go.Figure()
            for run_name in df_logs["run"].unique():
                sub = df_logs[df_logs["run"] == run_name]
                fig2.add_trace(go.Scatter(
                    x=sub["step"], y=sub["reward"],
                    mode="lines", name=run_name, line=dict(width=1.5),
                ))
            fig2.update_layout(
                title=f"Courbe d'entrainement — {result.get('_algo','')}",
                xaxis_title="Steps", yaxis_title="Recompense",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                height=320, margin=dict(l=20, r=20, t=36, b=20),
                legend=dict(font=dict(size=9)),
            )
            st.plotly_chart(fig2, width="stretch")

    elif result.get("_mode") == "reel":
        # ── Mode robot reel ───────────────────────────────────────────────────
        _mpath = result.get("_mpath", "")
        _mok   = os.path.exists(_mpath)
        _env_lbl = ENV_NAMES.get(result.get("_env", ""), result.get("_env", "—"))

        st.write(f"Algorithme : {result.get('_algo', '—')}")
        st.write(f"Environnement : {_env_lbl}")
        st.write(f"Modele : {'Disponible' if _mok else 'Absent'}")

        if _mok:
            st.info(
                "Pour executer sur le robot reel, lancez le script suivant dans un terminal :\n\n"
                "```bash\npython src/robot/sim_to_real.py\n```"
            )
        else:
            st.warning(
                f"Modele introuvable :\n`{_mpath}`\n\n"
                "Entrainez d'abord le modele avec les scripts dans `src/robot/`."
            )
