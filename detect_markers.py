import cv2
import numpy as np

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Load predefined dictionary of ArUco markers
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters_create()

    detected_markers = set()  # To track unique marker IDs
    marker_positions = []  # List to store positions of detected markers

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        # If markers are detected
        if ids is not None:
            frame_positions = []  # Store marker positions for the current frame
            for i, marker_id in enumerate(ids.flatten()):
                # Add unique marker ID to the set
                if marker_id not in detected_markers:
                    detected_markers.add(marker_id)
                    print(f"Detected Marker ID: {marker_id}")

                # Get the corner points of the marker
                corner_points = corners[i][0]  # Extract corner points
                center = np.mean(corner_points, axis=0)  # Compute center of the marker

                # Add marker ID, center, and corners to the frame_positions list
                frame_positions.append({
                    "id": marker_id,
                    "center": center,
                    "corners": corner_points
                })

                # Draw detected marker boundaries
                cv2.polylines(frame, [corner_points.astype(int)], True, (0, 255, 0), 2)

                # Display marker ID at its center
                center_int = center.astype(int)
                cv2.putText(frame, f"ID: {marker_id}", tuple(center_int), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Append the current frame's marker positions to the marker_positions list
            marker_positions.append(frame_positions)

        # Stop after detecting 5 unique markers
        if len(detected_markers) >= 5:
            print("Detected 5 markers. Stopping...")
            break

        # Display the video frame with detections
        cv2.imshow("ArUco Marker Detection", frame)

        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert marker positions to a NumPy array (optional)
    marker_positions_np = np.array(marker_positions, dtype=object)

    # Print all detected marker positions
    print("Detected Marker Positions (per frame):")
    for frame_index, frame_data in enumerate(marker_positions):
        print(f"Frame {frame_index + 1}:")
        for marker_data in frame_data:
            print(f"  ID: {marker_data['id']}, Center: {marker_data['center']}, Corners: {marker_data['corners']}")


    main()
