# SAE S5 - Détection et Localisation 3D avec RealSense

## 📋 Fichiers essentiels du projet

### Phase 1 : Préparation caméra

1. **`scan_cameras.py`** - Identifier les ID des caméras disponibles
   - Lance ce script pour trouver l'ID de la RealSense (ex: 37)
   
2. **`test_camera.py`** - Tester la caméra et capturer des vidéos
   - Vérifie que la caméra fonctionne
   - Peut enregistrer des vidéos de test
   - **Résolution harmonisée : 640x480** (résolution native RealSense)

3. **`test_depth_realsense.py`** - Visualiser RGB et heatmap de profondeur
   - Affiche l'image RGB et la heatmap de profondeur côte à côte
   - **Pédagogique :** Comprendre le fonctionnement RGB-D

4. **`generer_marqueurs_aruco.py`** - Générer les marqueurs ArUco
   - Crée une planche A4 paysage avec 4 marqueurs (IDs 3, 4, 5, 6)
   - Format : 300 DPI pour impression précise
   - Output : `aruco_markers_a4.png`
   - **Important :** Mesurer les distances réelles après impression

### Phase 2 : Calibration intrinsèque

5. **`calibration_intrinsique.py`** - Calibration custom avec damier
   - Capture ~20 images du damier sous différents angles
   - Calcule les paramètres intrinsèques (matrice K, distorsion)
   - Génère `calibration_intrinseque.pkl`
   - **Résolution : 640x480** (harmonisée avec RealSense)
   - **Utilité :** Comprendre la distorsion, corriger les images du dataset YOLO

6. **`get_realsense_intrinsics.py`** - Récupérer les paramètres d'usine RealSense
   - Extrait les intrinsics RGB et Depth depuis la caméra
   - Génère `realsense_calibration.json` (utilisé par comparaison_calibrations.py)
   - **Résolution : 640x480** (résolution native supportée)
   - **Utilité :** Affichage pédagogique + alimentation du script de comparaison

7. **`comparaison_calibrations.py`** - Comparer custom vs RealSense
   - **Dépend de :** `calibration_intrinseque.pkl` ET `realsense_calibration.json`
   - Affiche les différences entre les deux calibrations (focales, distorsion)
   - Visualisation en temps réel de l'impact des corrections
   - Génère un rapport d'analyse avec recommandations
   - **Pédagogique :** Comprendre pourquoi custom ≠ RealSense pour RGB-D

### Phase 3 : Création du dataset YOLO

**Dossier :** `dataset_localisation/`

8. **`enregistrer_video_corrigee.py`** - Enregistrer vidéo AVEC correction de distorsion
   - **S'inspire de test_camera.py** mais utilise `calibration_intrinseque.pkl`
   - Applique `cv2.undistort()` en temps réel pendant l'enregistrement
   - **Résolution : 640x480** (harmonisée avec calibration)
   - Enregistre la vidéo CORRIGÉE dans `dataset_localisation/video_corrigee_*.avi`
   - Commandes : 'r' démarrer/arrêter enregistrement, 's' capture image, 'q' quitter
   - **Avantage :** Vidéo directement corrigée, pas besoin de l'étape 10 sur les frames

9. **`extraire_frames.py`** - Extraire images depuis vidéo
   - Extrait 1 frame tous les N frames depuis une vidéo
   - **Redimensionne automatiquement à 640x480** (harmonisé RealSense)
   - Sortie : dossier `frames/` avec images numérotées
   - Usage : Créer dataset pour entraîner YOLO

10. **`renumeroter_images.py`** - Renuméroter après suppression
   - Renumérote les images dans l'ordre séquentiel
   - Utile après avoir supprimé des images floues/inutiles
   - Évite les trous dans la numérotation

11. **`Annotation des images`** - Label Studio
   - Annoter les objets sur les **images renumerotées**
   - Format YOLO : fichier `.txt` par image avec format `class x_center y_center width height`
   - Stocker dans `dataset_yolo/images/` et `dataset_yolo/labels/`
   - Créer `dataset_yolo/classes.txt` avec les noms des classes

12. **`detection_yolo.py`** - Entraîner YOLO sur dataset custom
   - Réorganise le dataset en train/val/test (70%/15%/15%)
   - Crée le fichier YAML de configuration
   - Entraîne YOLO11n pendant 50 epochs
   - Teste sur le test set
   - Sortie : `runs/detect/detection_objets/weights/best.pt`

### Phase 4 : Détection et localisation

13. **`visualiser_repere_camera.py`** - Visualiser les repères 3D
   - Affiche le repère caméra en 3D avec matplotlib
   - Visualise les transformations entre repères
   - **Pédagogique :** Comprendre les changements de repères (caméra → monde)

14. **`detection_avec_repere_aruco.py`** - Détection avec marqueur unique
   - Détecte le marqueur ArUco 6 (repère monde = centre du marqueur)
   - Détecte les objets avec YOLO
   - Localise en 3D dans le repère du marqueur 6
   - **Simple et rapide** : Un seul marqueur nécessaire
   - Moyennage 5×5 pixels pour profondeur stable
   - **Format bbox** : YOLO convertit automatiquement `center_x,center_y,w,h` normalisé → `x1,y1,x2,y2` pixels absolus via `results[0].boxes.xyxy`.

15. **`detection_avec_repere_aruco_map.py`** - **DÉTECTION ROBUSTE MULTI-MARQUEURS** (alternative à l'étape 14)
   - Détecte les marqueurs ArUco 3, 4, 5, 6 (configuration A4)
   - **Repère monde = centre de la feuille A4**
   - Calibration extrinsèque robuste (méthode Kabsch/SVD)
   - **Fonctionne avec minimum 3/4 marqueurs visibles**
   - Affiche l'erreur de calibration en temps réel
   - Demande les distances mesurées entre marqueurs
   - Moyennage 5×5 pixels pour profondeur stable
   - **Recommandé pour production** : Plus précis et résistant à l'occlusion

---

## 🔄 Ordre d'exécution

```bash
# PHASE 1 : Préparation caméra
# 1. Trouver l'ID de la caméra
python scan_cameras.py

# 2. Tester la caméra (optionnel)
python test_camera.py

# 3. Visualiser profondeur (optionnel, pédagogique)
python test_depth_realsense.py

# 4. Générer les marqueurs ArUco
python generer_marqueurs_aruco.py
# → Imprime aruco_markers_a4.png en A4 paysage
# → Mesure : taille marqueur + distances entre centres

# PHASE 2 : Calibration intrinsèque
# 5. Calibration custom (pour comprendre et corriger dataset YOLO)
python calibration_intrinsique.py

# 6. Récupérer intrinsics RealSense (génère realsense_calibration.json)
python get_realsense_intrinsics.py

# 7. Comparer les calibrations (pédagogique - nécessite étapes 5 et 6)
python comparaison_calibrations.py

# PHASE 3 : Dataset YOLO (optionnel - si besoin d'entraîner)
# 8. Enregistrer vidéo corrigée
python enregistrer_video_corrigee.py

# 9. Extraire frames depuis vidéo
cd dataset_localisation
python extraire_frames.py

# 10. Renuméroter après suppression d'images floues
python renumeroter_images.py

# 11. Annoter avec Label Studio
label-studio start
# → Importer frames/, annoter, exporter en YOLO format

# Organiser dans dataset_yolo/
# dataset_yolo/
#   ├── images/          # Images annotées
#   ├── labels/          # Fichiers .txt (1 par image)
#   └── classes.txt      # Liste des classes

# 12. Entraîner YOLO
cd ..
python detection_yolo.py

# PHASE 4 : Détection et localisation
# 13. Visualiser les repères 3D (optionnel, pédagogique)
python visualiser_repere_camera.py

# 14. Détection avec marqueur unique (simple)
python detection_avec_repere_aruco.py

# 15. Détection multi-marqueurs robuste (RECOMMANDÉ)
python detection_avec_repere_aruco_map.py
```

**💡 Pourquoi utiliser enregistrer_video_corrigee.py (étape 8) ?**
- Applique la correction de distorsion EN TEMPS RÉEL pendant l'enregistrement
- Vidéo MP4 déjà corrigée → frames extraites sont déjà sans distorsion
- Plus besoin d'étape séparée de correction après extraction

---

## 🎯 Quelle méthode de détection choisir ?

| Critère | `detection_avec_repere_aruco.py` | `detection_avec_repere_aruco_map.py` |
|---------|----------------------------------|--------------------------------------|
| **Marqueurs requis** | 1 seul (ID 6) | 4 marqueurs (IDs 3,4,5,6) |
| **Robustesse** | ❌ Si marqueur caché = pas de repère | ✅ Fonctionne avec 3/4 marqueurs |
| **Précision** | ⚠️ Dépend d'un seul marqueur | ✅ Calibration sur 4 points |
| **Repère monde** | Centre du marqueur 6 | Centre de la feuille A4 |
| **Mesures requises** | Taille du marqueur | Taille + distances entre marqueurs |
| **Setup** | 🟢 Rapide (1 marqueur) | 🟡 Nécessite impression A4 |
| **Erreur affichée** | ❌ Non | ✅ Oui (erreur de calibration en cm) |
| **Usage recommandé** | Tests rapides, démo | Production, précision maximale |

---

## 📚 Quelle calibration utiliser ?

| Cas d'usage | Calibration à utiliser | Fichier |
|-------------|------------------------|---------|
| **Corriger la distorsion du dataset YOLO** | Custom (damier) | `calibration_intrinseque.pkl` |
| **Localisation 3D avec RealSense** | RealSense (usine) | `realsense_calibration.json` |
| **Détection ArUco + localisation 3D** | RealSense (usine) | Directement via SDK |

### ⚠️ Important
- **Calibration custom** : Mesure la vraie distorsion optique (k2 ≈ -0.39)
  - Utile pour corriger les images avant annotation YOLO
  - **NE PAS utiliser pour la localisation 3D avec RealSense**
  
- **Intrinsics RealSense** : Calibration d'usine professionnelle
  - Utiliser pour `cv2.solvePnP` et déprojection 3D
  - Garantit la cohérence RGB ↔ Depth

### 🔗 Workflow de calibration
1. **calibration_intrinsique.py** → génère `calibration_intrinseque.pkl`
2. **get_realsense_intrinsics.py** → génère `realsense_calibration.json`
3. **comparaison_calibrations.py** → charge les 2 fichiers et compare
   - Affiche les différences de focales, distorsion, centre optique
   - Montre visuellement l'impact des corrections
   - Explique pourquoi custom ne fonctionne pas pour RGB-D

---

## 📂 Structure du projet

```
sae_s5_a_01_7/
├── dataset_localisation/
│   ├── extraire_frames.py          # Extraction frames (640x480)
│   ├── renumeroter_images.py       # Renumérotation images
│   ├── frames/                     # Images extraites
│   ├── video_corrigee_*.mp4        # Vidéo corrigée (sortie étape 8)
│   ├── output.mp4                  # Autres vidéos
│   └── dataset_yolo/
│       ├── images/                 # Images annotées
│       ├── labels/                 # Labels YOLO (.txt)
│       └── classes.txt             # Noms des classes
├── enregistrer_video_corrigee.py   # Enregistrement vidéo corrigée
├── detection_yolo.py               # Entraînement YOLO
├── generer_marqueurs_aruco.py      # Générateur marqueurs A4
├── calibration_intrinsique.py      # Calibration custom
├── get_realsense_intrinsics.py     # Intrinsics RealSense
├── comparaison_calibrations.py     # Comparaison calibrations
├── visualiser_repere_camera.py     # Visualisation 3D des repères
├── detection_avec_repere_aruco.py  # Détection 1 marqueur
├── detection_avec_repere_aruco_map.py  # Détection 4 marqueurs (ROBUSTE)
├── scan_cameras.py                 # Scanner caméras disponibles
├── test_camera.py                  # Test caméra basique
├── test_camera_simple.py           # Test caméra simplifié
├── test_depth_realsense.py         # Visualisation RGB + Depth
├── calibration_images/             # Images damier pour calibration
├── frames/                         # Frames extraites (racine)
├── frames_true/                    # Frames extraites (autre dossier)
├── output.mp4                      # Vidéos diverses
├── yolo11n.pt                      # Modèle YOLO pré-entraîné
├── aruco_markers_a4.png            # Planche 4 marqueurs A4
├── calibration_intrinseque.pkl     # Sortie calibration custom
├── realsense_calibration.json      # Sortie intrinsics RealSense
├── runs/detect/                    # Résultats YOLO
│   └── detection_objets/           # Modèle YOLO entraîné
│       └── weights/best.pt         # Meilleur modèle
└── ve_rl/                          # Environnement virtuel Python
```

---

## 🎓 Points pédagogiques clés

1. **Calibration monoculaire vs RGB-D**
   - Custom = vision monoculaire (1 caméra RGB)
   - RealSense = système RGB-D (RGB + Depth alignés)

2. **Distorsion optique**
   - La calibration custom mesure k2 = -0.39 (élevée)
   - RealSense corrige en interne → coefficients = 0

3. **Importance de la cohérence**
   - solvePnP et déprojection 3D doivent utiliser les **mêmes intrinsics**
   - Sinon : coordonnées Z incorrectes

4. **Transformations de repères**
   - `visualiser_repere_camera.py` aide à comprendre les axes X, Y, Z
   - Passage repère caméra → repère monde (marqueur ArUco)
   - Matrice de rotation + vecteur de translation

---
