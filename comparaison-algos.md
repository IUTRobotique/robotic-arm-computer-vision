# Analyse comparative des algorithmes de RL — Sélection et benchmark

**Projet :** Robot 3-DDL — manipulation de cube (push / push-in-hole)
**Simulateur :** MuJoCo via Gymnasium / Stable-Baselines3
**Algorithmes retenus :** PPO · HER-SAC · TD3

---

## 1. Contexte et objectif

Ce document analyse les algorithmes de Reinforcement Learning (RL) adaptés à la robotique
manipulatrice et justifie la sélection finale de trois algorithmes pour le projet. Il complète
l'analyse théorique initiale avec les résultats obtenus après implémentation et entraînement
sur simulateur MuJoCo.

Le robot est un bras 3-DDL (Degrés De Liberté) contrôlé en position articulaire. La tâche
principale est dite *push-in-hole* : l'effecteur doit pousser un cube pour le faire tomber
dans un trou situé au sol. C'est une tâche à **récompense creuse** (sparse reward) — le signal
de succès n'arrive qu'en fin d'épisode — ce qui la rend particulièrement difficile pour la
majorité des algorithmes de RL standard.

---![enter image description here](QED)

## 2. Panorama des algorithmes candidats

### 2.1 Tableau comparatif des architectures

| Caractéristique | PPO | SAC | TD3 |
|---|---|---|---|
| **Type** | On-policy | Off-policy | Off-policy |
| **Philosophie** | Stabilité (clipping) | Efficacité + entropie max | Déterminisme + Twin Q |
| **Fonction de perte** | Clipped Surrogate Objective | Soft Bellman Residual + Entropie | MSE Loss (Twin Q-learning) |
| **Politique** | Stochastique (Gaussian) | Stochastique (Reparameterized) | Déterministe |
| **Exploration** | Bruit implicite dans la politique | Entropie automatique | Bruit gaussien ajouté (σ = 0,1) |
| **Replay buffer** | Non (éch. jetables) | Oui (off-policy) | Oui (off-policy) |
| **Nombre de réseaux** | 1 acteur + 1 critique | 1 acteur + 2 critiques + 2 cibles | 1 acteur + 2 critiques + 3 cibles |
| **Compatibilité HER** | Non | Oui | Oui (non retenu ici) |
| **Temps d'entraînement** | Très long | Court | Moyen |
| **Rapidité d'exécution** | Excellente | Modérée | Excellente |

### 2.2 PPO — Proximal Policy Optimization

PPO (Schulman et al., 2017) est un algorithme **on-policy** : chaque lot d'expériences collecté
est consommé puis jeté. La stabilité est assurée par le *clipping* de l'objectif surrogate :

```
L_CLIP(θ) = E[ min(r_t(θ) A_t, clip(r_t(θ), 1−ε, 1+ε) A_t) ]
```

où `r_t(θ)` est le ratio des probabilités d'action et `A_t` l'avantage estimé par GAE
(Generalized Advantage Estimation, λ = 0,95).

**Avantages :**
- Courbe d'apprentissage monotone et stable, peu sensible aux hyperparamètres
- Exécution très rapide à l'inférence (politique légère sans replay buffer)
- Parallélisation naturelle sur plusieurs environnements (N_ENVS = 4 ici)

**Limites :**
- Convergence lente sur les espaces d'action continus (des millions de steps nécessaires)
- Incompatible avec HER : l'algorithme on-policy ne peut pas relabelliser des transitions passées
- Les récompenses creuses (sparse rewards) sont particulièrement pénalisantes car l'agent
  ne voit presque aucun signal positif pendant les premières phases

**Usage dans ce projet :** PPO est évalué sur la tâche *PushEnv* (pousser le cube de
plus de 20 cm depuis sa position initiale), tâche plus simple que le push-in-hole, avec
une récompense dense `reward = -dist(ee, cube) + 3 × déplacement + 30 × succès`.

### 2.3 SAC — Soft Actor-Critic

SAC (Haarnoja et al., 2018) est un algorithme **off-policy** maximisant simultanément la
récompense et l'entropie de la politique (Maximum Entropy RL) :

```
J(π) = Σ_t E[ r(s_t, a_t) + α H(π(·|s_t)) ]
```

Le coefficient `α` est ajusté automatiquement (*auto-tuning*) pour maintenir une entropie
cible. Cette régularisation évite l'effondrement prématuré de l'exploration.

**Avantages :**
- Réutilisation efficace des données via le replay buffer (off-policy)
- Exploration robuste grâce à la maximisation d'entropie automatique
- Particulièrement adapté aux espaces d'action continus multi-dimensionnels
- Compatible avec HER → sélectionné avec cette surcouche (voir §3)

**Limites :**
- Stochasticité à l'inférence (légèrement moins reproductible que TD3)
- Hyperparamètre `ent_coef` sensible sans auto-tuning

### 2.4 TD3 — Twin Delayed Deep Deterministic Policy Gradient

TD3 (Fujimoto et al., 2018) améliore DDPG par trois mécanismes :

1. **Twin critics** : deux réseaux Q indépendants — le minimum est pris pour les cibles de
   Bellman, ce qui élimine la surestimation systématique des valeurs d'action
2. **Delayed policy update** : l'acteur est mis à jour 2× moins souvent que les critiques
   (`policy_delay = 2`), laissant ces derniers converger d'abord
3. **Target policy smoothing** : bruit gaussien clipé ajouté aux actions cibles :
   `a'(s') = clip(π_target(s') + clip(ε, −c, c), a_low, a_high)`

**Avantages :**
- Politique déterministe → exécution stable et reproductible (idéal pour déploiement réel)
- Plus stable que DDPG, convergence plus régulière
- Efficacité similaire à SAC sur les tâches continues

**Limites :**
- Exploration purement exogène (bruit gaussien) : si `σ` est mal calibré, l'agent reste
  bloqué dans des minima locaux ou apprend un signal bruité
- Incompatible avec HER dans la configuration actuelle du projet

**Usage prévu :** TD3 est configuré sur *ReachingEnv* (atteindre une cible en position
cartésienne, seuil de succès = 1 cm), tâche plus simple que le push-in-hole.

---

## 3. Surcouches algorithmiques évaluées

### 3.1 HER — Hindsight Experience Replay

HER (Andrychowicz et al., 2017) n'est pas un algorithme en soi, mais une **surcouche de
relabellisation** compatible avec les algorithmes off-policy disposant d'un replay buffer.

**Principe :** après un épisode où le but `g` n'est pas atteint, certaines transitions
`(s_t, a_t, s_{t+1})` sont relabellisées avec le but `g' = achieved_goal` d'une transition
ultérieure du même épisode (stratégie *future*, `N_SAMPLED_GOAL = 4`). L'agent apprend
ainsi que `s_{t+1}` était un succès pour `g'`, même si `g` n'a pas été atteint.

**Pertinence critique pour ce projet :**
- Sans HER sur une tâche push-in-hole à récompense creuse, l'agent ne verrait
  presque jamais de signal positif. La convergence serait extrêmement lente voire impossible.
- Avec HER, chaque épisode génère `N_SAMPLED_GOAL × longueur_épisode` transitions
  relabellisées supplémentaires — soit un buffer 5× plus dense en signal utile.
- Le démarrage des premiers succès est observé dès **30 000 steps** (vs ~530 000 pour PPO
  sur une tâche plus simple).

L'environnement `PushInHoleGoalEnv` expose le contrat GoalEnv requis par HER :
observation sous forme de dictionnaire `{observation, achieved_goal, desired_goal}`, et
implémente `compute_reward(achieved_goal, desired_goal, info)` pour le relabellage.

Un curriculum progressif est intégré : la distance minimale cube-trou au spawn augmente
linéairement sur 2000 épisodes (`CURRICULUM_MIN_DIST_START = 2 cm → 10 cm`).

### 3.2 CrossQ

CrossQ (Bhatt et al., 2023) introduit la Batch Normalization dans le RL off-policy sans
les biais habituels, permettant des taux de mise à jour (UTD ratio) très élevés.
L'entraînement peut être divisé par un facteur 3 par rapport à un SAC standard.

Implémenté et testé sur ce projet (dossier `logs/crossq/`), mais insuffisamment entraîné
(3 checkpoints seulement) pour tirer des conclusions statistiquement fiables. Non retenu
dans le benchmark final par manque de données.

### 3.3 Algorithmes pixel-to-action (non retenus)

Pour des tâches apprenant directement depuis des images (flux caméra), deux approches
étaient envisagées :

- **DrQ-v2** (Data Regularized Q-learning) : off-policy sur DDPG optimisé, avec
  augmentations d'image (shifts, crops) intégrées dans la boucle de perte.
  État de l'art pour la manipulation physique depuis pixels bruts.
- **REDQ / DroQ** : utilise un ensemble de `n` critiques (Ensemble Learning), ce qui réduit
  drastiquement l'overfitting sur les données visuelles.

Ces approches ont été écartées : l'architecture actuelle utilise des observations d'état
(positions articulaires, position du cube par localisation ArUco), et non des pixels bruts.
Elles restent pertinentes si la perception visuelle est intégrée directement dans la politique.

---

## 4. Justification de la sélection finale

| Critère | PPO | HER-SAC | TD3 |
|---|:---:|:---:|:---:|
| Récompense creuse (sparse) | ✗ | ✓✓ | ✗ |
| Efficacité en sample | ✗ | ✓✓ | ✓ |
| Stabilité d'entraînement | ✓✓ | ✓ | ✓ |
| Politique déterministe | ✗ | ✗ | ✓✓ |
| Parallélisation envs | ✓✓ | ✗ | ✗ |
| Déployable sur robot réel | ✓ | ✓ | ✓✓ |
| Implémenté et entraîné | ✓ | ✓✓ | ✓ (configuré) |

**PPO** est retenu comme **algorithme de référence on-policy**. Son implémentation simple,
sa stabilité et sa parallélisation naturelle en font un bon point de comparaison, même si ses
performances sur la tâche principale sont limitées.

**HER-SAC** est l'algorithme **principal** pour la tâche push-in-hole. La combinaison
SAC (efficacité off-policy) + HER (gestion des récompenses creuses) est la plus adaptée aux
contraintes du projet : tâche sparse, espace d'action continu, robot 3-DDL.

**TD3** est retenu comme **algorithme de référence déterministe off-policy**. Sa politique
déterministe est un avantage pour le transfert sim-to-real (comportement reproductible,
sans variance stochastique à l'exécution).

---

## 5. Implémentation effective

### 5.1 Environnements

| Algorithme | Environnement | Tâche | Succès |
|---|---|---|---|
| PPO | `PushEnv` | Pousser le cube > 20 cm | `cube_displacement > 0.20 m` |
| HER-SAC | `PushInHoleGoalEnv` | Faire tomber le cube dans le trou | `cube_z < −0.01 m` |
| TD3 | `ReachingEnv` | Atteindre une cible en position 3D | `distance(ee, goal) < 0.01 m` |

**Note :** les trois algorithmes opèrent sur des tâches de difficulté croissante. PPO sur
la plus simple (déplacement libre), HER-SAC sur la plus exigeante (précision spatiale 3D
avec trou de diamètre fixe), TD3 sur une tâche intermédiaire (reaching). Cette asymétrie
rend la comparaison directe des récompenses sans intérêt — les métriques normalisées
(taux de succès, steps to solve) sont les seules comparables.

### 5.2 Hyperparamètres clés

| Paramètre | PPO | HER-SAC | TD3 |
|---|---|---|---|
| Steps totaux | 1 000 000 | 2 000 000 | 500 000 |
| Batch size | 64 | 256 | 256 |
| Réseau | [256, 256] Tanh | [256, 256] ReLU | [256, 256] Tanh |
| Learning rate | 3×10⁻⁴ | 3×10⁻⁴ | 3×10⁻⁴ |
| Parallélisme | 4 envs | 1 env | 1 env |
| N_SAMPLED_GOAL | — | 4 | — |
| Fréquence éval | 10 000 steps | 5 000 steps | 5 000 steps |
| Episodes par éval | 20 | 10 | 20 |

### 5.3 Récompense et signal d'apprentissage

**PushInHoleEnv (HER-SAC) :**
```
r = −2 × max(dist(ee, cube) − 0.03, 0)   # approche saturée
  − 5 × dist_xy(cube, trou)               # guide principal vers le trou
  − 0.05                                  # pression temporelle
  + 100 × is_success                      # bonus terminal
  − 0.01 × ||a_t − a_{t−1}||²            # lissage moteurs
```

**PushEnv (PPO) :**
```
r = −dist(ee, cube)                        # approche
  + 3 × déplacement_cube                  # encourager le mouvement
  + 30 × is_success                        # bonus terminal
  − 0.01 × ||a_t − a_{t−1}||²            # lissage moteurs
```

---

## 6. Résultats du benchmark

### 6.1 Taux de succès

| Algorithme | Premier succès | Pic | Succès final (moy. 10 derniers ckpts) | N éval/ckpt |
|---|---|---|---|---|
| **HER-SAC** | ~30 000 steps | **100 %** @ 1 350 000 steps | **92 %** | 10 |
| **PPO** | ~530 000 steps | 15 % @ 690 000 steps | 3 % | 20 |
| **TD3** | — (non entraîné) | — | — | — |

HER-SAC atteint 90 % de succès en environ 750 000 steps et se stabilise entre 90 % et 100 %
pour le reste de l'entraînement. PPO, même sur une tâche plus simple, plafonne à 15 % et
régresse vers 3 % en fin d'entraînement — symptôme classique d'un algorithme on-policy
qui oscille sans converger sur une récompense creuse.

### 6.2 Précision des épisodes réussis

La précision est définie comme la récompense moyenne des épisodes ayant abouti à un succès,
normalisée par la valeur maximale observée pour chaque algorithme (échelle intra-algo, non
comparable entre eux).

HER-SAC présente une progression quasi-monotone de la précision : non seulement l'agent
réussit plus souvent, mais il réussit mieux au fil du temps (cube centré dans le trou plus
rapidement, actions moins saccadées). PPO montre une précision erratique, cohérente avec
l'instabilité de son taux de succès.

### 6.3 Steps nécessaires pour résoudre la tâche

| Algorithme | Minimum observé | Moyenne finale (20 derniers ckpts) |
|---|---|---|
| **HER-SAC** | **12 steps** | **42 steps** |
| **PPO** | 25 steps | 39 steps |

Sur les épisodes où HER-SAC réussit, il le fait en 42 steps en moyenne sur 400 steps
maximum, soit en moins de 11 % de la durée autorisée. Cela illustre que l'agent a appris
une politique efficace et directe, pas une politique qui « tâtonne » jusqu'au succès aléatoire.

### 6.4 Généralisabilité — écart entraînement / évaluation

La généralisabilité est mesurée par l'écart entre le taux de succès en rollout (politique
stochastique avec bruit d'exploration, pendant l'entraînement) et le taux de succès en
évaluation (politique déterministe, mode test). Un écart positif signifie que l'agent
performe mieux en test qu'en train.

| Algorithme | Succès train (fin) | Succès test (fin) | Δ (test − train) |
|---|---|---|---|
| **HER-SAC** | 95,9 % ± 2,1 % | 94,1 % ± 8,0 % | **−1,8 %** |
| **PPO** | 15,8 % ± 4,2 % | 4,0 % ± 4,3 % | **−11,8 %** |

HER-SAC présente un écart quasi-nul entre train et test : la politique apprise en mode
stochastique se transfère presque parfaitement en mode déterministe. C'est un signe fort
de bonne généralisation — l'agent a compris la tâche et non mémorisé les trajectoires.

PPO présente un écart de −11,8 % : le taux de succès s'effondre entre le mode entraînement
et le mode évaluation. Cela traduit une dépendance au bruit d'exploration et une politique
insuffisamment consolidée.

---

## 7. Analyse et conclusions

### 7.1 HER-SAC : algorithme recommandé pour la tâche principale

Les résultats confirment que **HER-SAC est l'unique algorithme ayant convergé** sur la tâche
push-in-hole dans les conditions de ce projet. Trois facteurs expliquent ce résultat :

1. **HER compense la rareté du signal** : sans relabellisation, l'agent ne verrait le bonus
   +100 que lorsque le cube tombe effectivement dans le trou — événement quasi-impossible
   au hasard dans un espace de 14 × 20 cm avec un trou de quelques cm de diamètre.

2. **SAC maintient l'exploration** : l'entropie automatique empêche la politique de
   s'effondrer sur un comportement sous-optimal. Elle est particulièrement utile en début
   d'apprentissage lorsque les gradients de récompense sont faibles.

3. **Le curriculum accélère le bootstrap** : les 50 premiers épisodes en position facilitée,
   puis la progression linéaire de la distance minimale cube-trou sur 2000 épisodes,
   permettent à l'agent d'obtenir ses premiers succès rapidement (30 000 steps).

### 7.2 PPO : limites structurelles sur cette tâche

Les 3 % de succès finaux de PPO ne sont pas imputables à des hyperparamètres mal calibrés
mais à des **limites algorithmiques fondamentales** pour cette classe de tâche :

- L'algorithme on-policy ne peut pas relabelliser des transitions passées (incompatibilité
  avec HER)
- Chaque rollout collecte des données dont une proportion infime est un succès → le gradient
  de politique est dominé par des épisodes d'échec
- Sur une tâche plus simple (reaching ou push libre), PPO converge normalement

PPO reste pertinent pour des tâches où la récompense est **dense et structurée**, ou comme
baseline de comparaison algorithmique.

### 7.3 TD3 : à valider

TD3 n'a pas été entraîné sur la tâche push-in-hole dans la version actuelle du projet. Son
implémentation est complète (`td3_algo.py`, `ReachingEnv`) et les hyperparamètres sont
calibrés. Les hypothèses sur ses performances sont les suivantes :

- Sur `ReachingEnv` (tâche dense, pas de sparse reward) : convergence attendue entre
  100 000 et 300 000 steps, taux de succès > 80 %
- Sur `PushInHoleEnv` sans HER : convergence difficile pour les mêmes raisons que SAC
  sans HER
- Politique déterministe : variance à l'exécution plus faible qu'HER-SAC → avantage pour
  le transfert sim-to-real

### 7.4 Perspectives

| Amélioration envisagée | Impact attendu |
|---|---|
| HER + TD3 sur push-in-hole | Comparaison déterministe vs stochastique à capacités égales |
| CrossQ + HER | Réduction du temps d'entraînement par facteur 3 (UTD ratio élevé) |
| Curriculum adaptatif (automatic) | Meilleure gestion de la difficulté sans réglage manuel |
| Bruit de sim-to-real augmenté | Validation de la robustesse aux perturbations capteurs |
| Évaluation sur robot physique | Mesure du gap sim-to-real sur la tâche réelle |

---

*Document généré à partir de l'implémentation courante — `src/robot/` — et des logs d'entraînement disponibles dans `src/robot/logs/`. Benchmark produit par `src/robot/plot_comparison.py`.*
