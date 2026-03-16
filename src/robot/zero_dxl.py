import os
from dynamixel_sdk import *

# --- CONFIGURATION ---
DEVICE_NAME          = '/dev/ttyACM0'  # Modifier selon votre système (ex: 'COM3' sur Windows)
BAUDRATE             = 1000000
PROTOCOL_VERSION     = 2.0

ADDR_TORQUE_ENABLE   = 24
ADDR_GOAL_POSITION   = 30

TORQUE_ENABLE        = 1
# 512 correspond à 150° (le milieu de la course de 0 à 1023)
GOAL_POSITION_CENTER = 512 
DXL_IDS              = [1, 2, 3]

# --- INITIALISATION ---
portHandler = PortHandler(DEVICE_NAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("Erreur : Impossible d'ouvrir le port")
    quit()

print(f"Connexion établie. Alignement des moteurs sur 150° (valeur {GOAL_POSITION_CENTER})...")

# --- ACTION ---
try:
    for dxl_id in DXL_IDS:
        print(f"Configuration du moteur ID {dxl_id}...")
        # Activer le couple
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        
        # Envoyer l'ordre vers la position centrale (150°)
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
            portHandler, dxl_id, ADDR_GOAL_POSITION, GOAL_POSITION_CENTER
        )

except KeyboardInterrupt:
    print("\nInterruption utilisateur.")

finally:
    portHandler.closePort()
    print("Port fermé.")