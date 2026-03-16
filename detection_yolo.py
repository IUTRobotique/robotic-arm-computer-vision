"""
Détection d'objets avec YOLO
=============================

Script pour entraîner et tester YOLO sur un dataset personnalisé.
Basé sur le cours: melodiedaniel.github.io/deep_learning/chap6_partie2.html

Étapes:
1. Préparer le dataset au format YOLO
2. Créer le fichier de configuration YAML
3. Entraîner YOLO
4. Tester sur des images

Utilisation:
    python detection_yolo.py
"""

import os
import yaml
import shutil
from pathlib import Path
import torch
from torch.utils.data import random_split

def prepare_yolo_dataset(source_dir, output_dir='data_yolo', seed=42, train_ratio=0.70, val_ratio=0.15):
    """
    Réorganise le dataset pour YOLO (train/val/test split).
    
    Args:
        source_dir: Dossier source contenant images/ et labels/
        output_dir: Dossier de sortie pour la structure YOLO
        seed: Seed pour reproductibilité
        train_ratio: Proportion du train set
        val_ratio: Proportion du val set
    
    Returns:
        Path vers output_dir, liste des classes
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    images_dir = source_dir / 'images'
    labels_dir = source_dir / 'labels'
    classes_file = source_dir / 'classes.txt'
    
    # Vérifier que les dossiers existent
    if not images_dir.exists():
        print(f"❌ Dossier images non trouvé: {images_dir}")
        return None, []
    
    if not labels_dir.exists():
        print(f"❌ Dossier labels non trouvé: {labels_dir}")
        return None, []
    
    if not classes_file.exists():
        print(f"❌ Fichier classes.txt non trouvé: {classes_file}")
        return None, []
    
    print(f"🔄 Préparation du dataset YOLO depuis: {source_dir}")
    
    # Créer la structure de dossiers
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Récupérer toutes les images
    image_files = sorted(list(images_dir.glob('*.jpg')) + 
                        list(images_dir.glob('*.jpeg')) + 
                        list(images_dir.glob('*.png')))
    
    print(f"📁 {len(image_files)} images trouvées")
    
    if len(image_files) == 0:
        print("❌ Aucune image trouvée !")
        return None, []
    
    # Créer le split
    total_size = len(image_files)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_indices, val_indices, test_indices = random_split(
        range(len(image_files)),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"📊 Split (seed={seed}): {train_size} train, {val_size} val, {test_size} test")
    
    # Copier les fichiers
    splits = {
        'train': train_indices.indices,
        'val': val_indices.indices,
        'test': test_indices.indices
    }
    
    for split_name, indices in splits.items():
        print(f"\n📂 Préparation du split '{split_name}'...")
        
        for idx in indices:
            img_file = image_files[idx]
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            # Copier l'image
            dest_img = output_dir / 'images' / split_name / img_file.name
            shutil.copy(img_file, dest_img)
            
            # Copier le label
            if label_file.exists():
                dest_label = output_dir / 'labels' / split_name / label_file.name
                shutil.copy(label_file, dest_label)
    
    # Copier classes.txt
    shutil.copy(classes_file, output_dir / 'classes.txt')
    
    # Charger les classes
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print(f"\n📋 Classes ({len(classes)}): {classes}")
    print(f"\n✅ Dataset YOLO préparé dans: {output_dir}")
    
    return output_dir, classes


def create_yolo_yaml(output_dir, classes, yaml_filename='dataset.yaml'):
    """
    Crée le fichier YAML de configuration pour YOLO.
    
    Args:
        output_dir: Dossier racine du dataset
        classes: Liste des noms de classes
        yaml_filename: Nom du fichier YAML
    
    Returns:
        Path vers le fichier YAML
    """
    output_dir = Path(output_dir)
    yaml_path = output_dir / yaml_filename
    
    config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Fichier YAML créé: {yaml_path}")
    return yaml_path


def train_yolo(yaml_path, model_name='yolo11n.pt', epochs=50, batch=8, project='runs/detect', name='detection_objets'):
    """
    Entraîne YOLO sur le dataset.
    
    Args:
        yaml_path: Chemin vers le fichier YAML
        model_name: Modèle YOLO à utiliser
        epochs: Nombre d'epochs
        batch: Taille du batch
        project: Dossier de sortie
        name: Nom de l'expérience
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics non installé. Installation...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print("ENTRAÎNEMENT DE YOLO")
    print(f"{'='*60}")
    print(f"Modèle: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Dataset: {yaml_path}")
    print(f"{'='*60}\n")
    
    # Charger le modèle
    model = YOLO(model_name)
    
    # Entraîner
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=0 if torch.cuda.is_available() else 'cpu',
        project=project,
        name=name,
        patience=10
    )
    
    print(f"\n✅ Entraînement terminé!")
    print(f"📁 Résultats dans: {project}/{name}")
    
    return model


def test_yolo(model_path, yaml_path, save_dir='runs/detect/test_predictions'):
    """
    Teste YOLO sur le test set.
    
    Args:
        model_path: Chemin vers le modèle entraîné
        yaml_path: Chemin vers le fichier YAML
        save_dir: Dossier pour sauvegarder les prédictions
    """
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print("TEST DE YOLO SUR LE TEST SET")
    print(f"{'='*60}")
    
    # Charger le modèle
    model = YOLO(model_path)
    
    # Évaluer sur le test set
    metrics = model.val(data=str(yaml_path), split='test')
    
    print(f"\n📊 Métriques sur le TEST SET:")
    print(f"  mAP@0.5     : {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"  Precision   : {metrics.box.mp:.3f}")
    print(f"  Recall      : {metrics.box.mr:.3f}")
    
    print(f"\n✅ Test terminé!")


def main():
    """
    Fonction principale pour la détection d'objets avec YOLO.
    """
    print("\n" + "="*60)
    print("DÉTECTION D'OBJETS AVEC YOLO")
    print("="*60)
    
    # Étape 1: Préparer le dataset
    print(f"\n{'='*60}")
    print("ÉTAPE 1: PRÉPARATION DU DATASET")
    print(f"{'='*60}")
    
    source_dir = './dataset_localisation/dataset_yolo'
    output_dir = './data_yolo'
    
    output_path, classes = prepare_yolo_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        seed=42
    )
    
    if output_path is None:
        print("\n❌ Échec de la préparation du dataset")
        return
    
    # Étape 2: Créer le fichier YAML
    print(f"\n{'='*60}")
    print("ÉTAPE 2: CRÉATION DU FICHIER YAML")
    print(f"{'='*60}")
    
    yaml_path = create_yolo_yaml(output_path, classes)
    
    # Étape 3: Entraîner YOLO
    print(f"\n{'='*60}")
    print("ÉTAPE 3: ENTRAÎNEMENT DE YOLO")
    print(f"{'='*60}")
    
    model = train_yolo(
        yaml_path=yaml_path,
        model_name='yolo11n.pt',
        epochs=50,
        batch=8,
        project='runs/detect',
        name='detection_objets'
    )
    
    # Étape 4: Tester
    print(f"\n{'='*60}")
    print("ÉTAPE 4: TEST SUR LE TEST SET")
    print(f"{'='*60}")
    
    best_model_path = 'runs/detect/detection_objets/weights/best.pt'
    test_yolo(best_model_path, yaml_path)
    
    print(f"\n{'='*60}")
    print("✅ DÉTECTION D'OBJETS TERMINÉE!")
    print(f"{'='*60}")
    print("\nFichiers générés:")
    print(f"  - {output_dir}/ : Dataset YOLO réorganisé")
    print(f"  - {yaml_path} : Configuration YOLO")
    print(f"  - runs/detect/detection_objets/ : Résultats d'entraînement")
    print(f"  - {best_model_path} : Meilleur modèle")


if __name__ == "__main__":
    main()
