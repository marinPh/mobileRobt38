#!/usr/bin/env python
# Project: 		ArUco Marker Detector
# Date created: 	12/18/2021
# Python version: 	3.8
# Reference: 		https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
# Following Tutorial:	https://automaticaddison.com/how-to-detect-aruco-markers-using-opencv-and-python/

from __future__ import print_function  	# Python 2/3 compatibility
import cv2  				# Import the OpenCV library
import numpy as np  			# Import Numpy library
import math 				# for arctan
import sys  				# Import sys library
import queue



def main(qpos: queue.Queue, qimg: queue.Queue):
  """
  Main method of the program.
  """
  
  
# Specify the ArUco dictionary
  desired_aruco_dictionary = "DICT_6X6_250"

# Check if the dictionary is valid
  ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
  }

  if desired_aruco_dictionary not in ARUCO_DICT:
      print(f"[ERROR] ArUco tag type '{desired_aruco_dictionary}' is not supported")
      exit(1)
  # Check that we have a valid ArUco marker
  if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
      print("[INFO] ArUCo tag of '{}' is not supported".format(
      args["type"]))
      sys.exit(0)

  # Load the dictionary
  print(f"[INFO] detecting '{desired_aruco_dictionary}' markers...")
  this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
  this_aruco_parameters = cv2.aruco.DetectorParameters()
  

  # Start the video stream
  cap = cv2.VideoCapture(0)
  
  # Set resolution
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the width
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the height
  
  # Corners of the ArUco square (initialize as empty)
  square_corners = []
      
  # Corner Map Positions
  tag1 = (0, 0) # in case none detected
  tag2 = (0, 0) # in case none detected
  tag3 = (0, 0) # in case none detected
  tag4 = (0, 0) # in case none detected

  while(True):

    # Capture frame-by-frame
    # This method returns True/False as well as the video frame
    _, frame = cap.read()
    

    # Resize the frame to avoid cropping wrong aspect ratio
    _ = cv2.resize(frame, (1280, 720))

    # Detect ArUco markers in the video frame
    (corners, ids, _) = cv2.aruco.detectMarkers(
      frame, this_aruco_dictionary, parameters=this_aruco_parameters)

    # Check that at least one ArUco marker was detected
    if len(corners) > 0:
      # Flatten the ArUco IDs list
      ids = ids.flatten()

      # Loop over the detected ArUco corners
      for (marker_corner, marker_id) in zip(corners, ids):

        # Extract the marker corners
        corners = marker_corner.reshape((4, 2))
        (top_left, top_right, bottom_right, bottom_left) = corners

        # Convert the (x,y) coordinate pairs to integers
        top_right = (int(top_right[0]), int(top_right[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        top_left = (int(top_left[0]), int(top_left[1]))

        # Draw the bounding box of the ArUco detection
        cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
        cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
        cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
        cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

        # Calculate and draw the center of the ArUco marker
        center_x = int((top_left[0] + bottom_right[0]) / 2.0)
        center_y = int((top_left[1] + bottom_right[1]) / 2.0)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        # Draw the ArUco marker ID on the video frame
        # The ID is always located at the top_left of the ArUco marker
#        cv2.putText(frame, str(marker_id), (top_left[0], top_left[1] - 15),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	
        ## DRAW BLUE OUTLINE AROUND MAP ---------------------------------------
        # Note all 4 corners of the map
        if marker_id == 1:
          tag1 = top_right
          pos1 = (center_x, center_y) 
        if marker_id == 2:
          tag2 = bottom_left
        if marker_id == 3:
          tag3 = bottom_left
        if marker_id == 4:
          tag4 = bottom_left
        if marker_id == 5:
          pos5 = (center_x, center_y) 
          corner5A = top_left
          corner5B = bottom_left
    
    # Drawing lines between all centers of Corner tags
    cv2.line(frame, tag1, tag2, (255, 0, 0), 2)     
    cv2.line(frame, tag2, tag3, (255, 0, 0), 2)     
    cv2.line(frame, tag3, tag4, (255, 0, 0), 2)     
    cv2.line(frame, tag4, tag1, (255, 0, 0), 2) 

    ## WARP PERSPECTIVE -------------------------------------------------------
    # Not paid enough to do this properly
    # for now we do a taccone!

    # Collect all corners of the square formed by ArUco markers
#    square_corners = np.array([tag1,tag2,tag3,tag4])
  
    # Define output corners
#    output_corners = np.array([[0,0],[1280,0],[1280,720],[0,720]])

    # Compute the perspective transform matrix
#    matrix = cv2.getPerspectiveTransform(square_corners, output_corners)
    
    # Apply the perspective warp
#    normalized_image = cv2.warpPerspective(frame, matrix, (1280, 720))

# add normalized_image to queue
#    qimg.put(normalized_image)

    ## COMPUTE (X,Y,YAW) OF TAG #5 --------------------------------------------
    # Scaling map to mm 
    width = tag2[0]-tag1[0]
    height = tag3[1]-tag2[1]
    #TODO: REMEASURE
    real_width = 800		#mm  --> REMEASURE, I DIDNT HAVE RULER
    real_height = 400		#mm  --> REMEASURE, I DIDNT HAVE RULER
    #TODO: must be visible in the frame otherwise div by 0
    #TODO: must have a verification that all 5 tags are visible in the frame
    x_scale = real_width/width
    y_scale = real_height/height

    # Computing #5 Yaw
    dx = corner5A[0] - corner5B[0]
    dy = corner5A[1] - corner5B[1]
    yaw5 = math.atan2(dx, dy) 
    yaw5 = math.degrees(yaw5)
 
    # Compute Tag #5's Position
    scaled_pos5 = ( ( pos5[0] - tag1[0] ) * x_scale, 
                    ( pos5[1] - tag1[1] ) * y_scale,
                    ( yaw5 ) )
    
    print(scaled_pos5)
    print("\n")

    # Draw Tag #5 Position on Frame
    cv2.putText(frame, 
                f"(X: {int(scaled_pos5[0])}, Y: {int(scaled_pos5[1])}, YAW: {int(scaled_pos5[2])})", 
                (pos5[0], pos5[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 2)
    qpos.put(scaled_pos5)
    
    
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # If "q" is pressed on the keyboard, 
    # exit this loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Close down the video stream
  cap.release()
  cv2.destroyAllWindows()


