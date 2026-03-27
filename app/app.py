"""YoloBliss — Page unique : algo -> env -> mode -> resultats."""
import glob
import os
import subprocess
import sys

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from utils.paths import ALGO_INFO, ENV_NAMES, MAIN_SCRIPT, ROBOT_SRC, model_path, log_path  # noqa
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
    """Lit les logs TensorBoard SB3.
    Retourne {"reward": DataFrame, "success": DataFrame|None} ou None.
    - reward  : eval/mean_reward  > rollout/ep_rew_mean
    - success : eval/success_rate > rollout/success_rate
    Seul le run le plus recent non-vide est utilise.
    """
    import glob
    import pandas as pd
    ldir = log_path(algo_key)
    if not os.path.isdir(ldir):
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    runs = sorted(
        [r for r in glob.glob(os.path.join(ldir, "*")) if os.path.isdir(r)],
        key=os.path.getmtime,
        reverse=True,
    )
    if not runs:
        return None
    for run in runs:
        try:
            ea = EventAccumulator(run, size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            rew_tag = next((t for t in ("eval/mean_reward", "rollout/ep_rew_mean", "train/reward") if t in tags), None)
            if rew_tag is None:
                continue
            df_rew = pd.DataFrame(
                [{"step": e.step, "value": e.value} for e in ea.Scalars(rew_tag)]
            ).sort_values("step")
            suc_tag = next((t for t in ("eval/success_rate", "rollout/success_rate") if t in tags), None)
            df_suc = None
            if suc_tag:
                df_suc = pd.DataFrame(
                    [{"step": e.step, "value": e.value} for e in ea.Scalars(suc_tag)]
                ).sort_values("step")
            return {"reward": df_rew, "success": df_suc}
        except Exception:
            continue
    return None


def _render_training_charts(algo_key: str, height: int = 380) -> bool:
    """Affiche les courbes d'entrainement (recompense + taux de succes).
    Retourne True si des donnees ont ete trouvees.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    logs = _read_tb_logs(algo_key)
    if logs is None:
        return False

    df_rew = logs["reward"]
    df_suc = logs["success"]
    has_success = df_suc is not None and not df_suc.empty

    ncols = 2 if has_success else 1
    subplot_titles = ["Recompense moyenne (lissee)"]
    if has_success:
        subplot_titles.append("Taux de succes")

    fig = make_subplots(rows=1, cols=ncols, subplot_titles=subplot_titles)

    smooth_rew = df_rew["value"].rolling(10, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df_rew["step"], y=df_rew["value"],
        mode="lines", line=dict(width=1, color="rgba(76,184,255,0.25)"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_rew["step"], y=smooth_rew,
        mode="lines", name="Recompense", line=dict(width=2.5, color="#4cb8ff"),
        showlegend=False,
    ), row=1, col=1)

    if has_success:
        smooth_suc = df_suc["value"].rolling(5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df_suc["step"], y=df_suc["value"],
            mode="lines", line=dict(width=1, color="rgba(74,222,128,0.25)"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=df_suc["step"], y=smooth_suc,
            mode="lines", name="Succes", line=dict(width=2.5, color="#4ade80"),
            showlegend=False,
        ), row=1, col=2)
        fig.update_yaxes(range=[0, 1], tickformat=".0%", row=1, col=2)

    fig.update_layout(
        title=f"Entrainement — {algo_key}",
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(title_text="Steps")
    fig.update_yaxes(title_text="Recompense", row=1, col=1)
    if has_success:
        fig.update_yaxes(title_text="Taux de succes", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)
    return True


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
    algo_labels = [
        f"{name}  (pas de modele entraine)" if not os.path.exists(model_path(name)) else name
        for name in algo_names
    ]
    default_algo_index = algo_names.index(st.session_state.algo) if st.session_state.algo in algo_names else 0
    algo_label = st.selectbox("Algorithme", algo_labels, index=default_algo_index)
    algo_choice = algo_names[algo_labels.index(algo_label)]
    if algo_choice != st.session_state.algo:
        _pick_algo(algo_choice)

    st.write(ALGO_INFO[st.session_state.algo]["description"])

    current_model_path = model_path(st.session_state.algo)
    if os.path.exists(current_model_path):
        st.success("Modele disponible")
    else:
        st.warning("Modele introuvable : la simulation utilisera une politique aleatoire.")

    has_model = os.path.exists(current_model_path)

    st.markdown("---")

    # ── Boutons Réel / Simulé ─────────────────────────────────────────────
    btn_reel, btn_sim = st.columns(2, gap="medium")
    disabled_cls = " disabled" if not has_model else ""

    with btn_reel:
        st.markdown(
            f'<div class="mode-btn reel{disabled_cls}">'
            '<div class="icon"></div>'
            '<div class="label">Réel</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        reel_clicked = st.button(
            "Lancer Réel" if has_model else "Pas de modèle",
            key="btn_reel",
            use_container_width=True,
            disabled=not has_model,
        )

    with btn_sim:
        st.markdown(
            '<div class="mode-btn sim">'
            '<div class="icon"></div>'
            '<div class="label">Simulé</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        sim_clicked = st.button(
            "Lancer Simulation",
            key="btn_sim",
            use_container_width=True,
        )

    if reel_clicked:
        st.session_state.mode = "Robot Reel"
        st.session_state.running = True
        st.session_state.result = None
        st.rerun()
    if sim_clicked:
        st.session_state.mode = "Simulation"
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

        _main_algo = ALGO_INFO[algo_key]["main_algo"]
        _cmd_base  = [sys.executable, MAIN_SCRIPT,
                      "--env", env_key, "--algo", _main_algo, "--render"]

        if mode_key == "Simulation":
            cmd = _cmd_base
            with st.spinner(f"Simulation lancee — {algo_key}  /  {ENV_NAMES.get(env_key, env_key)} …"):
                try:
                    subprocess.Popen(cmd, cwd=ROBOT_SRC)
                    res = {"_mode": "sim", "_algo": algo_key, "_env": env_key,
                           "_cmd": " ".join(cmd)}
                except Exception as e:
                    res = {"error": str(e), "_mode": "sim"}
        else:
            cmd = _cmd_base + ["--real"]
            with st.spinner(f"Robot reel lance — {algo_key}  /  {ENV_NAMES.get(env_key, env_key)} …"):
                try:
                    subprocess.Popen(cmd, cwd=ROBOT_SRC)
                    res = {"_mode": "reel", "_algo": algo_key, "_env": env_key,
                           "_cmd": " ".join(cmd)}
                except Exception as e:
                    res = {"error": str(e), "_mode": "reel"}

        st.session_state.result  = res
        st.session_state.running = False
        st.rerun()

    # ── Affichage ─────────────────────────────────────────────────────────────
    result = st.session_state.result

    if result is None:
        # Etat initial : afficher courbes de logs si algo deja choisi
        selected_algo = st.session_state.algo
        if selected_algo:
            found = _render_training_charts(selected_algo, height=420)
            if not found:
                st.info("Aucun log d'entrainement disponible pour cet algorithme.")
                st.write("Appuyez sur Lancer pour demarrer une simulation.")
        else:
            st.info("Choisissez un algorithme, un environnement, un mode, puis appuyez sur Lancer.")

    elif "error" in result:
        st.error(result["error"])

    elif result.get("_mode") == "sim":
        _env_lbl = ENV_NAMES.get(result.get("_env", ""), result.get("_env", "—"))
        st.success(f"Simulation lancee : {result.get('_algo', '')}  /  {_env_lbl}")
        st.code(result.get("_cmd", ""), language="bash")
        _render_training_charts(result.get("_algo", ""), height=320)

    elif result.get("_mode") == "reel":
        _env_lbl = ENV_NAMES.get(result.get("_env", ""), result.get("_env", "—"))
        _mok = os.path.exists(model_path(result.get("_algo", "")))
        st.success(f"Commande robot reel : {result.get('_algo', '')}  /  {_env_lbl}")
        st.code(result.get("_cmd", ""), language="bash")
        if not _mok:
            st.warning("Modele introuvable — le robot utilisera une politique aleatoire.")
        _render_training_charts(result.get("_algo", ""), height=320)
