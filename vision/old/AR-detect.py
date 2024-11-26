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
import numpy as np

# ======================= GLOBAL VARIABLES ====================================


# ========================== FUNCTIONS ========================================



# =============================== MAIN ========================================
# Load the predefined dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)  # You can choose another dictionary

# Adjusti Detection parameters
parameters = aruco.DetectorParameters_create()

# Start the video capture
cap = cv2.VideoCapture(0) # On my computer USB Camera is on port /dev/video4

while True:
	ret, frame = cap.read()
	if not ret:
		break

	# Convert the frame to grayscale (required for AR detection)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray) 

	# Detect markers
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

	# Draw detected markers
	if ids is not None:
		aruco.drawDetectedMarkers(frame, corners, ids)

	# Display the resulting frame
	cv2.imshow('AR Tag Detection', frame)

	
	# Print debug information
	print("Corners detected:", corners)
	print("IDs detected:", ids)


	# Break the loop on 'q' key press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()



