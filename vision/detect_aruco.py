#!/usr/bin/env python
# Project: ArUco Marker Detector
# Python version: 3.8
import cv2
import numpy as np
import math
import sys
import queue
import json

desired_aruco_dictionary = "DICT_ARUCO_ORIGINAL"

# The different ArUco dictionaries built into the OpenCV library
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
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

def main(channel: queue.Queue,stop:queue.Queue):
    with open('./vision/camera_calibration.json', 'r') as file:
        params = json.load(file)
    
    camera_matrix = np.array(params["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(params["distortion_coefficients"], dtype=np.float32)

    

    
    """
    Main method of the program.
    """
    if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
        print(f"[INFO] ArUCo tag of '{desired_aruco_dictionary}' is not supported")
        sys.exit(0)

    print(f"[INFO] Detecting '{desired_aruco_dictionary}' markers...")
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[desired_aruco_dictionary])
    aruco_params = cv2.aruco.DetectorParameters_create()

    # Start the video stream
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize ArUco square corners
    square_corners = []
    tag_positions = {1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (0, 0)}

    while True and stop.empty():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video")
            break

        frame = cv2.resize(frame, (1280, 720))

        h,  w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))

        # undistort
        dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        frame = dst[y:y+h, x:x+w]

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        if len(corners) >= 4:
            ids = ids.flatten()
            for marker_corner, marker_id in zip(corners, ids):
                marker_corner = marker_corner.reshape((4, 2))
                top_left, top_right, bottom_right, bottom_left = [tuple(map(int, corner)) for corner in marker_corner]

                # Draw bounding box
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                # Draw center
                center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                if marker_id in tag_positions:
                    tag_positions[marker_id] = (center_x, center_y)

            # Warp perspective if all 4 markers are detected
            if all(tag_positions.values()):
                square_corners = np.array(
                    [tag_positions[1], tag_positions[2], tag_positions[3], tag_positions[4]]
                ).astype(np.float32)

                output_corners = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]]).astype(np.float32)
                matrix = cv2.getPerspectiveTransform(square_corners, output_corners)
                normalized_image = cv2.warpPerspective(frame, matrix, (1280, 720))

                # Display results
                query = (True, normalized_image, normalized_image, (0, 0, 0))
                try:
                    channel.put(query, timeout=0.1)
                except queue.Full:
                    continue

        else:
            query = (False, frame, frame, (0, 0, 0))
            try:
                channel.put(query, timeout=0.1)
            except queue.Full:
                continue

        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
