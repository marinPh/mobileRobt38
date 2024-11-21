# ================================ HEADER =====================================
# Author:       Loic Delineau
# Date:         21/11/2024
# Licence:     	GNU-GPLv3 
# File:        	AR-detect.py 
# Platform :    Any Ubuntu machine
# Description:	Detects AR tags on map and normalises it

# ============================ LIBRARIES ======================================
import cv2
import cv2.aruco as aruco
import os

# ========================== GENERATE SCRIPT ==================================
# Set the dictionary to 5x5
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100) 

# Define the output folder to save marker images
output_folder = "aruco_markers"
os.makedirs(output_folder, exist_ok=True)

# Define marker size in pixels
marker_size = 200  # e.g., 200x200 pixels

# Generate and save markers
for marker_id in range(aruco_dict.bytesList.shape[0]):
    marker_image = 255 - aruco.drawMarker(aruco_dict, marker_id, marker_size)
    
    # Save the marker image
    filename = os.path.join(output_folder, f"marker_{marker_id}.png")
    cv2.imwrite(filename, marker_image)
    print(f"Saved marker ID {marker_id} as {filename}")

print("All markers have been generated and saved.")


