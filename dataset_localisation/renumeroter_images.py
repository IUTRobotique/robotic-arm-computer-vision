"""
Renumérotation des images dans l'ordre
=======================================

Script pour renuméroter les images dans l'ordre séquentiel
après suppression de certaines images.

Utilisation:
    python renumeroter_images.py
"""

import os
import shutil

def renumeroter_images(images_dir='frames'):
    """
    Renumérote les images dans l'ordre séquentiel.
    
    Args:
        images_dir: Dossier contenant les images
    """
    if not os.path.exists(images_dir):
        print(f"❌ Dossier {images_dir} non trouvé")
        return
    
    # Lister tous les fichiers images
    fichiers = []
    for f in os.listdir(images_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            fichiers.append(f)
    
    if len(fichiers) == 0:
        print(f"❌ Aucune image trouvée dans {images_dir}")
        return
    
    print(f"\n📋 Trouvé {len(fichiers)} images à renuméroter")
    print(f"Dossier: {images_dir}")
    print(f"Les images seront renommées de frame_00000 à frame_{len(fichiers)-1:05d}\n")
    
    # Renommer directement avec des noms temporaires d'abord
    print("🔄 Renumérotation en cours...")
    
    # Étape 1: Renommer avec des noms temporaires pour éviter les conflits
    temp_names = []
    for i, ancien_nom in enumerate(fichiers):
        extension = os.path.splitext(ancien_nom)[1]
        temp_nom = f'temp_{i:05d}{extension}'
        
        ancien_chemin = os.path.join(images_dir, ancien_nom)
        temp_chemin = os.path.join(images_dir, temp_nom)
        
        os.rename(ancien_chemin, temp_chemin)
        temp_names.append(temp_nom)
    
    # Étape 2: Renommer avec les noms finaux
    for i, temp_nom in enumerate(temp_names):
        extension = os.path.splitext(temp_nom)[1]
        nouveau_nom = f'frame_{i:05d}{extension}'
        
        temp_chemin = os.path.join(images_dir, temp_nom)
        nouveau_chemin = os.path.join(images_dir, nouveau_nom)
        
        os.rename(temp_chemin, nouveau_chemin)
        
        if i % 50 == 0 or i < 5:
            print(f"  Renommé: {nouveau_nom}")
    
    print(f"\n✅ RENUMÉROTATION TERMINÉE!")
    print(f"   {len(fichiers)} images renommées de frame_00000 à frame_{len(fichiers)-1:05d}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RENUMÉROTATION DES IMAGES")
    print("="*60)
    
    renumeroter_images('frames')
