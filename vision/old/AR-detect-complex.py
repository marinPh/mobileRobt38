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
parameters.adaptiveThreshWinSizeMin = 5
parameters.adaptiveThreshWinSizeMax = 33
parameters.adaptiveThreshWinSizeStep = 5
parameters.minMarkerPerimeterRate = 0.02  # Adjust for small markers
parameters.maxMarkerPerimeterRate = 4.5  # Adjust for large markers
parameters.polygonalApproxAccuracyRate = 0.01  # Tolerate imperfect edges
parameters.minCornerDistanceRate = 0.05  # Minimum corner spacing

# Start the video capture
cap = cv2.VideoCapture(0) # On my computer USB Camera is on port /dev/video4


while True:
	ret, frame = cap.read()
	if not ret:
		break

	# Convert the frame to grayscale (required for AR detection)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray) 
	_, gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

	# Detect markers
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

	# Draw detected markers
	if ids is not None:
		aruco.drawDetectedMarkers(frame, corners, ids)

	# Display the resulting frame
	cv2.imshow('AR Tag Detection', frame)
	cv2.imshow('Grayscale Frame', gray)

	
	# Print debug information
	print("Corners detected:", corners)
	print("IDs detected:", ids)


	for point in rejectedImgPoints:
    		cv2.polylines(frame, [point.astype(int)], True, (0, 0, 255), 2)
	cv2.imshow('Rejected Points', frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	# Break the loop on 'q' key press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()



