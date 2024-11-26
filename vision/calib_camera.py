import numpy as np
import cv2 as cv
import glob
import json

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0)
objp = np.zeros((7 * 10, 3), np.float32)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob("./vision/checkerboard_images/*")

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Failed to load image {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Debug: Show the grayscale image
    cv.imshow("Gray Image", gray)
    cv.waitKey(500)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (10, 7), None)

    print(f"Processing {fname}, Chessboard found: {ret}")

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (10, 7), corners2, ret)
        cv.imshow("Chessboard Corners", img)
        cv.waitKey(500)
    else:
        # Debug: Show the image where corners are not found
        cv.imshow("Corners Not Found", img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Calibrate the camera if we have enough points
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Check if calibration was successful
    if not ret:
        print("Camera calibration failed")
    else:
        print("Camera calibration was successful")

    # Print the calibration results
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    print("Rotation vectors:\n", rvecs)
    print("Translation vectors:\n", tvecs)

    # Save the calibration results to a JSON file
    calibration_data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "rotation_vectors": [rvec.tolist() for rvec in rvecs],
        "translation_vectors": [tvec.tolist() for tvec in tvecs],
    }

    with open("./vision/camera_calibration.json", "w") as json_file:
        json.dump(calibration_data, json_file, indent=4)

    print("Calibration data saved to camera_calibration.json")
else:
    print("Not enough points to calibrate the camera.")
