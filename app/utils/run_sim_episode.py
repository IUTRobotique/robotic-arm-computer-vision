"""Script autonome : lance un episode de simulation MuJoCo.

Usage:
    python run_sim_episode.py <env_name> <model_path|none> <output_dir> [max_steps]
"""
from __future__ import annotations
import json, os, sys

ROBOT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "robot"))
sys.path.insert(0, ROBOT_SRC)
sys.path.insert(0, os.path.join(ROBOT_SRC, "robot_env"))


def _write(output_dir: str, data: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(data, f, indent=2)


def _make_env(env_name: str):
    mapping = {
        "reaching":     ("robot_env.reaching_env",    "ReachingEnv"),
        "push":         ("robot_env.push_env",         "PushEnv"),
        "push_in_hole": ("robot_env.push_in_hole_env", "PushInHoleEnv"),
        "sliding":      ("robot_env.sliding_env",      "SlidingEnv"),
        "sorting":      ("robot_env.sorting_env",      "SortingEnv"),
    }
    if env_name not in mapping:
        return None
    mod_name, cls_name = mapping[env_name]
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        return cls(render_mode="rgb_array")
    except Exception as e:
        print(f"[run_sim] Env error: {e}", flush=True)
        return None


def _load_model(model_path: str | None, env):
    if not model_path or model_path == "none" or not os.path.exists(model_path):
        return None
    try:
        from stable_baselines3 import SAC, PPO, TD3
        for cls in (SAC, PPO, TD3):
            try:
                return cls.load(model_path, env=env)
            except Exception:
                continue
        try:
            from sbx import CrossQ
            return CrossQ.load(model_path, env=env)
        except Exception:
            pass
    except Exception as e:
        print(f"[run_sim] Model load error: {e}", flush=True)
    return None


def run_episode(env_name: str, model_path: str | None, output_dir: str, max_steps: int = 300) -> dict:
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    env = _make_env(env_name)
    if env is None:
        result = {"error": f"Environnement inconnu ou non installé : {env_name}"}
        _write(output_dir, result)
        return result

    model = _load_model(model_path, env)
    obs, _ = env.reset()
    total_reward, step = 0.0, 0
    terminated = truncated = False
    frames: list[np.ndarray] = []
    info: dict = {}

    for step in range(max_steps):
        action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame)
        if terminated or truncated:
            break

    env.close()

    # Sauvegarde video
    video_path = ""
    if frames:
        try:
            import cv2
            h, w = frames[0].shape[:2]
            video_path = os.path.join(output_dir, "episode.mp4")
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
        except Exception as e:
            print(f"[run_sim] Video error: {e}", flush=True)
            video_path = ""

    metrics = {
        "n_steps":      step + 1,
        "total_reward": round(total_reward, 4),
        "is_success":   bool(info.get("is_success", False)),
        "n_frames":     len(frames),
        "video_path":   video_path,
        "env":          env_name,
        "model":        model_path or "random",
    }
    _write(output_dir, metrics)
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run_sim_episode.py <env> <model|none> <output_dir> [max_steps]")
        sys.exit(1)
    env_arg    = sys.argv[1]
    model_arg  = sys.argv[2]
    out_arg    = sys.argv[3]
    steps_arg  = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    result = run_episode(env_arg, model_arg, out_arg, steps_arg)
    print(json.dumps(result, indent=2))
