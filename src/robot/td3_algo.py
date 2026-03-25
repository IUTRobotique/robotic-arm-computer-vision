"""Entraînement TD3 (Twin Delayed Deep Deterministic Policy Gradient).

TD3 (Fujimoto et al., 2018) améliore DDPG par trois mécanismes correctifs :

  1. Twin critics (Double Q-learning) : deux réseaux Q indépendants,
     on prend le minimum lors du calcul des cibles de Bellman pour éviter
     la surestimation systématique des valeurs d'action.

  2. Delayed policy update : l'acteur et les réseaux cibles sont mis à jour
     moins fréquemment que les critiques (tous les ``policy_delay`` steps).
     Ceci laisse les critiques converger avant de mettre à jour la politique.

  3. Target policy smoothing : du bruit gaussien clipé est ajouté aux actions
     cibles lors du calcul des cibles de Bellman :
         a'(s') = clip(π_target(s') + clip(ε, -c, c), a_low, a_high)
     Ceci régularise les critiques et réduit leur variance.

La politique est déterministe ; l'exploration se fait en ajoutant un bruit
gaussien N(0, σ²) à l'action lors de la collecte (pas lors de l'évaluation).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from robot_env.push_in_hole_env import PushInHoleEnv as ReachingEnv
from stable_baselines3.common.monitor import Monitor


TOTAL_TIMESTEPS: int = 300_000
BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 10_000
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005
LEARNING_RATE: float = 3e-4
GRADIENT_STEPS: int = 16

#l'acteur est mis à jour 1 fois pour 2 mises à jour des critiques
POLICY_DELAY: int = 2

N_ENVS: int = 8

#écart-type du bruit gaussien ajouté à l'action pendant la collecte
ACTION_NOISE_STD: float = 0.1

POLICY_KWARGS: dict[str, object] = {
    "net_arch": [256, 256],
    "activation_fn": torch.nn.Tanh,
}

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "td3")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "td3")

#récompense moyenne par épisode au-delà de laquelle l'entraînement s'arrête
#reaching : 100 steps × ~1.0 (succès) - petites pénalités ≈ 95 max ; 90 = tâche résolue
REWARD_THRESHOLD: float = 30.0  # FIX: abaisser le seuil d'arret TD3 selon la nouvelle cible


class _RenderCallback(BaseCallback):
    """Appelle training_env.render() à chaque pas de collecte.

    SB3 ne déclenche pas le rendu automatiquement pendant learn() ;
    ce callback est le seul moyen d'afficher la simulation en temps réel.
    """

    def _on_step(self) -> bool:
        self.training_env.render("human")
        return True


def make_env(render_mode: str | None =None) -> ReachingEnv:
    """Crée une instance de ReachingEnv (factory pour make_vec_env).
    Parameters:
        render_mode (str | None): ``"human"`` pour afficher MuJoCo en temps réel,
            ``None`` pour l'entraînement headless (plus rapide).
    Returns:
        ReachingEnv: environnement gymnasium initialisé.
    """
    return ReachingEnv(render_mode=render_mode)

def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    model_dir: str = MODEL_DIR,
    log_dir: str = LOG_DIR,
    render: bool = False,
) -> TD3:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    render_mode: str | None = "human" if render else None
    n_envs_train: int = 1 if render else N_ENVS
    vec_env_cls = None if n_envs_train == 1 else SubprocVecEnv

    env: VecEnv = make_vec_env(
        make_env,
        n_envs=n_envs_train,
        env_kwargs={"render_mode": render_mode},
        vec_env_cls=vec_env_cls,
        monitor_kwargs={"info_keywords": ("is_success", "cube_displacement")},
    )
    eval_env: VecEnv = make_vec_env(
        make_env,
        n_envs=1,
        env_kwargs={"render_mode": None},
        monitor_kwargs={"info_keywords": ("is_success", "cube_displacement")},
    )

    # Le bruit porte sur les dimensions d'action, pas sur n_envs
    # SB3 applique un bruit indépendant par env automatiquement
    n_actions: int = env.action_space.shape[0]
    action_noise: NormalActionNoise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=ACTION_NOISE_STD * np.ones(n_actions),
    )

    model: TD3 = TD3(
        "MlpPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_rate=LEARNING_RATE,
        gradient_steps=-1,          # auto : 1 gradient step par transition collectée
        policy_delay=POLICY_DELAY,
        action_noise=action_noise,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=log_dir,
        verbose=1,
    )

    n_params: int = sum(p.numel() for p in model.policy.parameters())
    print(f"Device: cpu | n_envs_train: {n_envs_train} | vec_env: {type(env).__name__} | Paramètres : {n_params:,}")

    stop_callback: StopTrainingOnRewardThreshold = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THRESHOLD,
        verbose=1,
    )
    eval_callback: EvalCallback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(5_000 // n_envs_train, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    callbacks: list[BaseCallback] = [eval_callback]
    if render:
        callbacks.append(_RenderCallback())

    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")

    model.save(os.path.join(model_dir, "td3_final"))
    env.close()
    eval_env.close()
    return model

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Entraînement TD3 sur le robot 3-DDL (MuJoCo)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Nombre de pas d'environnement (défaut : {TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Affiche la simulation MuJoCo en temps réel pendant l'entraînement"
    )
    args: argparse.Namespace = parser.parse_args()
    train(total_timesteps=args.timesteps, render=args.render)
