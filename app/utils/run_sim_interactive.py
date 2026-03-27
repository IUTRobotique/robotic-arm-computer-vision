"""Script autonome : lance un episode MuJoCo en mode interactif (fenetre viewer).

Utilise render_mode='human' pour ouvrir la fenetre MuJoCo interactive.
L'utilisateur peut interagir avec la vue (rotation, pause, etc.).
A la fin de l'episode, les metriques sont ecrites dans output_dir/metrics.json.

Usage:
    python run_sim_interactive.py <env_name> <main_algo> <model_path|none> <output_dir> [max_steps]

    main_algo : sac | ppo | crossq | her
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

ROBOT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "robot"))
sys.path.insert(0, ROBOT_SRC)
sys.path.insert(0, os.path.join(ROBOT_SRC, "robot_env"))


ALGO_CLS = {
    "sac":    "SAC",
    "ppo":    "PPO",
    "td3":    "TD3",
    "crossq": "SAC",
    "her":    "SAC",
}

STEP_DELAY = 1 / 25  # ~25 fps — laisse le temps a la fenetre de se rafraichir
FPS = 25


def _write(output_dir: str, data: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(data, f, indent=2)


def _make_env(env_name: str, algo: str):
    """Cree l'env avec render_mode='human'."""
    if algo == "her":
        if env_name == "push_in_hole":
            from her_push_in_hole import PushInHoleGoalEnv
            return PushInHoleGoalEnv(render_mode="human")
        elif env_name == "sorting":
            from her_sorting import SortingGoalEnv
            return SortingGoalEnv(render_mode="human")
        else:
            return None

    mapping = {
        "reaching":     ("robot_env.reaching_env",    "ReachingEnv"),
        "push":         ("robot_env.push_env",         "PushEnv"),
        "sliding":      ("robot_env.sliding_env",      "SlidingEnv"),
        "push_in_hole": ("robot_env.push_in_hole_env", "PushInHoleEnv"),
        "sorting":      ("robot_env.sorting_env",      "SortingEnv"),
    }
    if env_name not in mapping:
        return None
    mod_name, cls_name = mapping[env_name]
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        return cls(render_mode="human")
    except Exception as e:
        print(f"[run_sim_interactive] Env error: {e}", flush=True)
        return None


def _load_model(model_path_str: str | None, algo: str, env):
    if not model_path_str or model_path_str == "none" or not os.path.exists(model_path_str):
        return None
    cls_name = ALGO_CLS.get(algo, "SAC")
    try:
        from stable_baselines3 import SAC, PPO, TD3
        cls_map = {"SAC": SAC, "PPO": PPO, "TD3": TD3}
        if cls_name in cls_map:
            return cls_map[cls_name].load(model_path_str, env=env)
    except Exception as e:
        print(f"[run_sim_interactive] Model load error: {e}", flush=True)
    return None


def _capture_frame(env) -> np.ndarray | None:
    """Capture une frame RGB via le renderer off-screen de gymnasium MuJoCo."""
    try:
        renderer = getattr(env.unwrapped, "mujoco_renderer", None)
        if renderer is not None:
            frame = renderer.render("rgb_array")
            if isinstance(frame, np.ndarray):
                return frame
    except Exception:
        pass
    return None


def _save_video(frames: list, output_dir: str) -> str:
    """Encode les frames en video MP4 (H.264 via ffmpeg si dispo, sinon WebM VP8)."""
    if not frames:
        return ""
    try:
        import cv2, shutil, subprocess as sp
        h, w = frames[0].shape[:2]
        if shutil.which("ffmpeg"):
            raw = os.path.join(output_dir, "episode_raw.mp4")
            final = os.path.join(output_dir, "episode.mp4")
            out = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
            ret = sp.run(
                ["ffmpeg", "-y", "-i", raw, "-c:v", "libx264",
                 "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", final],
                capture_output=True, timeout=120,
            )
            if ret.returncode == 0:
                os.remove(raw)
                return final
            os.rename(raw, final)
            return final
        else:
            path = os.path.join(output_dir, "episode.webm")
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"VP80"), FPS, (w, h))
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
            return path
    except Exception as e:
        print(f"[run_sim_interactive] Video error: {e}", flush=True)
        return ""


def _extract_distance(info: dict) -> float:
    for key in ("distance", "cube_displacement", "dist_cube_hole",
                "dist_cube_goal", "dist_cylinder_goal"):
        if key in info:
            return float(info[key])
    return float("nan")


def run_interactive(env_name: str, algo: str, model_path_str: str | None,
                    output_dir: str, max_steps: int = 300) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    env = _make_env(env_name, algo)
    if env is None:
        result = {"error": f"Environnement '{env_name}' non supporte (algo={algo})"}
        _write(output_dir, result)
        return result

    model = _load_model(model_path_str, algo, env)
    obs, _ = env.reset()
    total_reward, step = 0.0, 0
    info: dict = {}
    frames: list[np.ndarray] = []

    for step in range(max_steps):
        action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        env.render()
        frame = _capture_frame(env)
        if frame is not None:
            frames.append(frame)
        time.sleep(STEP_DELAY)
        if terminated or truncated:
            break

    # Episode termine — garder la fenetre ouverte jusqu'a fermeture manuelle
    print("[run_sim_interactive] Episode termine. Fermez la fenetre MuJoCo pour continuer.", flush=True)
    try:
        viewer = getattr(getattr(env.unwrapped, "mujoco_renderer", None), "viewer", None)
        if viewer is not None and hasattr(viewer, "is_running"):
            while viewer.is_running():
                env.render()
                frame = _capture_frame(env)
                if frame is not None:
                    frames.append(frame)
                time.sleep(STEP_DELAY)
        else:
            while True:
                env.render()
                frame = _capture_frame(env)
                if frame is not None:
                    frames.append(frame)
                time.sleep(STEP_DELAY)
    except Exception:
        pass  # La fenetre a ete fermee par l'utilisateur

    dist = _extract_distance(info)
    env.close()

    video_path = _save_video(frames, output_dir)

    metrics = {
        "n_steps":      step + 1,
        "total_reward": round(total_reward, 4),
        "is_success":   bool(info.get("is_success", False)),
        "distance":     round(dist, 4) if not (dist != dist) else None,
        "n_frames":     len(frames),
        "video_path":   video_path,
        "env":          env_name,
        "algo":         algo,
        "model":        model_path_str or "random",
    }
    _write(output_dir, metrics)
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: run_sim_interactive.py <env> <main_algo> <model|none> <output_dir> [max_steps]")
        sys.exit(1)
    env_arg   = sys.argv[1]
    algo_arg  = sys.argv[2]
    model_arg = sys.argv[3]
    out_arg   = sys.argv[4]
    steps_arg = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    result = run_interactive(env_arg, algo_arg, model_arg, out_arg, steps_arg)
    print(json.dumps(result, indent=2))
