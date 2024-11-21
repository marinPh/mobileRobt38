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
import numpy as np

# ========================== GENERATE SCRIPT ==================================
# Set the dictionary to 5x5
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)  # 5x5 markers, up to 100 markers

# Define the output folder to save marker images
output_folder = "aruco_markers_with_borders"
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

# Define marker size in pixels and border parameters
marker_size = 200  # Size of the AR tag in pixels
border_size = 20   # Thickness of the black border around the tag
white_padding = 50  # Thickness of the white background padding

# Generate and save markers with borders
for marker_id in range(aruco_dict.bytesList.shape[0]):
    # Create the AR marker
    marker_image = aruco.drawMarker(aruco_dict, marker_id, marker_size)
    
    # Add a black border
    marker_with_border = cv2.copyMakeBorder(
        marker_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=0  # Black color
    )
    
    # Add white padding around the black border
    marker_with_white = cv2.copyMakeBorder(
        marker_with_border,
        top=white_padding,
        bottom=white_padding,
        left=white_padding,
        right=white_padding,
        borderType=cv2.BORDER_CONSTANT,
        value=255  # White color
    )
    
    # Save the marker image
    filename = os.path.join(output_folder, f"marker_{marker_id}.png")
    cv2.imwrite(filename, marker_with_white)
    print(f"Saved marker ID {marker_id} as {filename}")

print("All markers with borders and padding have been generated and saved.")



