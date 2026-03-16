"""
Visualisation du repère de la caméra RealSense
===============================================

Ce script affiche le repère 3D de la caméra sur l'image en temps réel.
"""

import pyrealsense2 as rs
import numpy as np
import cv2


def draw_camera_frame_on_image(image, intrinsics, scale=0.1):
    """
    Dessine le repère 3D de la caméra sur l'image.
    
    Le repère caméra :
    - Origine : centre optique (centre de l'image en projection)
    - X (ROUGE)  : vers la droite
    - Y (VERT)   : vers le bas
    - Z (BLEU)   : profondeur (vers l'avant, sort de l'écran)
    
    Args:
        image: Image sur laquelle dessiner
        intrinsics: Paramètres intrinsèques de la caméra
        scale: Longueur des axes en mètres
    """
    # Origine du repère = centre optique projeté au centre de l'image
    origin_3d = [0, 0, scale * 5]  # À 50cm de profondeur pour visualisation
    
    # Points 3D des extrémités des axes (dans le repère caméra)
    # Axe X : de (0,0,Z) à (scale,0,Z)
    x_axis_3d = [scale, 0, scale * 5]
    
    # Axe Y : de (0,0,Z) à (0,scale,Z)
    y_axis_3d = [0, scale, scale * 5]
    
    # Axe Z : de (0,0,Z) à (0,0,Z+scale)
    z_axis_3d = [0, 0, scale * 5 + scale]
    
    # Projeter les points 3D en pixels 2D
    origin_2d = rs.rs2_project_point_to_pixel(intrinsics, origin_3d)
    x_end_2d = rs.rs2_project_point_to_pixel(intrinsics, x_axis_3d)
    y_end_2d = rs.rs2_project_point_to_pixel(intrinsics, y_axis_3d)
    z_end_2d = rs.rs2_project_point_to_pixel(intrinsics, z_axis_3d)
    
    # Convertir en entiers
    origin = tuple(map(int, origin_2d))
    x_end = tuple(map(int, x_end_2d))
    y_end = tuple(map(int, y_end_2d))
    z_end = tuple(map(int, z_end_2d))
    
    # Dessiner les axes
    # X en ROUGE
    cv2.arrowedLine(image, origin, x_end, (0, 0, 255), 3, tipLength=0.3)
    cv2.putText(image, "X (droite)", (x_end[0] + 10, x_end[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Y en VERT
    cv2.arrowedLine(image, origin, y_end, (0, 255, 0), 3, tipLength=0.3)
    cv2.putText(image, "Y (bas)", (y_end[0] + 10, y_end[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Z en BLEU
    cv2.arrowedLine(image, origin, z_end, (255, 0, 0), 3, tipLength=0.3)
    cv2.putText(image, "Z (profondeur)", (z_end[0] + 10, z_end[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Origine
    cv2.circle(image, origin, 8, (255, 255, 255), -1)
    cv2.putText(image, "Origine (0,0,0)", (origin[0] + 10, origin[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


def main():
    """
    Affiche le repère de la caméra en temps réel.
    """
    print("\n" + "="*60)
    print("VISUALISATION DU REPÈRE CAMÉRA REALSENSE")
    print("="*60)
    
    print("\n📐 REPÈRE CAMÉRA:")
    print("  Origine : Centre optique de la caméra RGB")
    print("  X (ROUGE)  → Vers la DROITE")
    print("  Y (VERT)   → Vers le BAS")
    print("  Z (BLEU)   → PROFONDEUR (vers l'avant)")
    
    print("\n💡 Sur l'image vous verrez:")
    print("  - Flèche ROUGE (X)  : direction droite")
    print("  - Flèche VERTE (Y)  : direction bas")
    print("  - Flèche BLEUE (Z)  : direction avant (raccourcie car projection)")
    
    print("\n⌨️  Appuyez sur 'q' pour quitter\n")
    
    # Initialiser RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    
    profile = pipeline.start(config)
    
    # Récupérer les intrinsèques
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    print("✅ Caméra initialisée")
    print(f"   Centre optique (cx, cy) = ({intrinsics.ppx:.1f}, {intrinsics.ppy:.1f}) pixels\n")
    
    # Stabiliser
    for _ in range(10):
        pipeline.wait_for_frames()
    
    try:
        while True:
            # Acquérir les frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convertir en numpy
            color_image = np.asanyarray(color_frame.get_data())
            
            # Dessiner le repère
            annotated = draw_camera_frame_on_image(color_image.copy(), intrinsics, scale=0.15)
            
            # Ajouter des explications
            y_offset = 30
            cv2.putText(annotated, "REPERE CAMERA REALSENSE", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_offset += 30
            cv2.putText(annotated, "Origine = Centre optique", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Afficher
            cv2.imshow('Repere Camera - Appuyez sur q pour quitter', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n✅ Arrêté")


if __name__ == "__main__":
    main()
