"""Utilitaires partagés entre les scripts d'entraînement.

Fournit ``HParamsCallback`` pour l'enregistrement des hyperparamètres dans
le plugin HParams de TensorBoard, permettant la comparaison visuelle entre
différents runs via l'onglet «HParams» du dashboard.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class HParamsCallback(BaseCallback):
    """Enregistre les hyperparamètres dans le plugin HParams de TensorBoard.

    À appeler en fin d'entraînement : écrit le dictionnaire d'hyperparamètres
    et la récompense finale dans le fichier d'événements TensorBoard du run courant.
    Dans TensorBoard, ouvrir l'onglet «HParams» pour comparer les runs entre eux
    selon n'importe quelle combinaison de paramètres.

    Note d'implémentation : ``SummaryWriter.add_hparams`` crée un sous-répertoire
    parasite. On contourne ce bug en écrivant directement dans le file_writer.
    Parameters:
        hparam_dict (dict[str, Any]): hyperparamètres à enregistrer (gamma, lr, etc.)
        verbose (int): niveau de verbosité SB3
    """

    hparam_dict: dict[str, Any]
    """Hyperparamètres à enregistrer dans l'onglet HParams."""

    def __init__(self, hparam_dict: dict[str, Any], verbose: int =0) -> None:
        super().__init__(verbose)
        self.hparam_dict = hparam_dict

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        """Écrit les hparams et la récompense finale dans le fichier TensorBoard."""
        tb_fmt: TensorBoardOutputFormat | None = next(
            (f for f in self.logger.output_formats
             if isinstance(f, TensorBoardOutputFormat)),
            None,
        )
        if tb_fmt is None:
            return

        #récompense moyenne sur le dernier buffer d'épisodes comme métrique de comparaison
        metric_dict: dict[str, float] = {"hparam/ep_rew_mean": 0.0}
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            metric_dict["hparam/ep_rew_mean"] = float(
                np.mean([float(ep["r"]) for ep in self.model.ep_info_buffer])
            )

        #écriture directe dans le file_writer pour rester dans le répertoire du run courant
        from torch.utils.tensorboard.summary import hparams as tb_hparams
        exp, ssi, sei = tb_hparams(self.hparam_dict, metric_dict)
        writer = tb_fmt.writer
        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)
        writer.flush()
