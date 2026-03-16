"""
Extraction de frames depuis une vidéo
======================================

Script simple pour extraire des images d'une vidéo.

Utilisation:
    python extraire_frames.py
"""

import cv2
import os

def extraire_frames_espacees(video_path, output_dir, intervalle=10, resize_to=(640, 480)):
    """
    Extrait 1 frame tous les N frames et redimensionne à la résolution cible.
    
    Args:
        video_path: chemin vers la vidéo
        output_dir: dossier de sortie
        intervalle: extraire 1 frame tous les N frames (ex: 10)
        resize_to: tuple (largeur, hauteur) pour redimensionner (défaut: 640x480 pour RealSense)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Erreur : impossible d'ouvrir {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Extraction de 1 frame tous les {intervalle} frames")
    if resize_to:
        print(f"   Redimensionnement : {resize_to[0]}x{resize_to[1]} (RealSense native)")
    print(f"   Total attendu : ~{total_frames // intervalle} images")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sauvegarder seulement toutes les N frames
        if frame_count % intervalle == 0:
            # Redimensionner à la résolution cible (640x480 pour RealSense)
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to)
            
            output_path = os.path.join(output_dir, f'frame_{saved_count:05d}.jpg')
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"✓ {saved_count} frames extraites sur {frame_count} totales")
    print(f"📁 Images sauvegardées dans: {output_dir}")


if __name__ == "__main__":
    # Configuration
    video_path = 'video_corrigee_20260122_181040.mp4'  # Nom de la vidéo dans le même dossier
    output_dir = 'frames_true'       # Dossier de sortie pour les images
    intervalle = 10             # Extraire 1 frame toutes les 10 frames
    
    print("\n" + "="*60)
    print("EXTRACTION DE FRAMES DEPUIS UNE VIDÉO")
    print("="*60)
    print(f"\nVidéo: {video_path}")
    print(f"Dossier de sortie: {output_dir}")
    print(f"Intervalle: 1 frame toutes les {intervalle} frames")
    print(f"Résolution: 640x480 (harmonisée RealSense)")
    print("="*60 + "\n")
    
    # Exemples de calcul d'intervalle:
    # - Vidéo à 30 fps, 1 frame/seconde → intervalle=30
    # - Vidéo à 30 fps, 1 frame toutes les 10 frames → intervalle=10
    # - Extraire ~100 images d'une vidéo de 900 frames → intervalle=9
    
    extraire_frames_espacees(video_path, output_dir, intervalle)
    
    print(f"\n✅ Extraction terminée!")
