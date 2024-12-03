import cv2
import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Mouse callback to get points
points = []

def select_points(event, x, y, flags, param):
    """
    Mouse callback function to capture clicks for selecting the goal point.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 1:  # Allow only one point
        print(f"Goal point selected at: ({x}, {y})")  # Debug print
        points.append((y, x))  # Append (row, col) in grid terms
        # Display the click on the image
        temp_image = param.copy()
        cv2.circle(
            temp_image, (x, y), 5, (0, 0, 255), -1
        )  # Draw a red dot for the click
        cv2.imshow("Select Goal", temp_image)


def create_costmap(image, grid_rows, grid_cols):
    """
    Discretize the image into a costmap and display the binary image.
    """
    # Threshold the image
    _, binary_image = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)
    binary_image = gaussian_filter(binary_image, sigma=20)
    plt.imshow(binary_image)

    height, width = image.shape
    block_height = height // grid_rows
    block_width = width // grid_cols
    
    reshaped = binary_image.reshape(grid_rows, block_height,  grid_cols, block_width)
    averages = np.mean(reshaped, axis=(1, 3))

    # Generate the costmap based on the average threshold
    costmap = np.where(averages > 200, 0, 1)
    
    recreated_binary_image = np.kron(
        (1 - costmap).astype(np.uint8), np.ones((block_height, block_width), dtype=np.uint8)
    ) * 255
    
    plt.imsave("costmap.png", recreated_binary_image, cmap='gray')


    return costmap, block_height, block_width



def heuristic(a, b):
    """
    Heuristic function for A* algorithm (Manhattan distance).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(costmap, start, goal):
    """
    A* algorithm for shortest path.
    Saves the costmap at each step for debugging purposes.
    """
    rows, cols = costmap.shape
    open_set = []
    heappush(open_set, (0, 0, start))  # (f_cost, g_cost, position)
    came_from = {}
    g_costs = {start: 0}
    explored = set()

    def heuristic(a, b):
        # Using Manhattan distance as heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(node):
        x, y = node
        x= int(x)
        y = int(y)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and costmap[nx, ny] == 0:
                yield (nx, ny)

    step = 0
    while open_set:
        _, current_g_cost, current_pos = heappop(open_set)
        
        explored.add(current_pos)

        # Save the current costmap for debugging
        #plt.imsave(f'costmap_step_{step}.png', costmap, cmap='gray')
        step += 1

        if current_pos == goal:
            path = []
            while current_pos in came_from:
                path.append(current_pos)
                current_pos = came_from[current_pos]
            path.append(start)
            return path[::-1]  # Reverse path

        for neighbor in neighbors(current_pos):
            tentative_g_cost = g_costs[current_pos] + 1
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                came_from[neighbor] = current_pos
                g_costs[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic(neighbor, goal)
                heappush(open_set, (f_cost, tentative_g_cost, neighbor))

    return []  # No path found

def path_pix_to_cm(path, block_width, block_height, cm_per_pixel):
    path_cm = []
    if path:
        for row, col in path:
            # Convert grid coordinates to pixel center
            center_x_pixels = (col + 0.5) * block_width
            center_y_pixels = (row + 0.5) * block_height

            # Convert pixels to cm
            center_x_cm = center_x_pixels * cm_per_pixel
            center_y_cm = center_y_pixels * cm_per_pixel

            # Append the point in cm
            path_cm.append((center_x_cm, center_y_cm))
    return path_cm

def path_visualization(frame, path, block_width, block_height):
    overlay_image = frame.copy()
    if path:
        path_centers_pixels = []
        for row, col in path:
            # Calculate the center of the grid cell in pixels
            center_x_pixels = int((col + 0.5) * block_width)
            center_y_pixels = int((row + 0.5) * block_height)
            path_centers_pixels.append((center_x_pixels, center_y_pixels))

            # Draw the center point
            cv2.circle(overlay_image, (center_x_pixels, center_y_pixels), 5, (0, 255, 0), -1)

        # Draw lines connecting the centers
        for i in range(len(path_centers_pixels) - 1):
            cv2.line(
                overlay_image,
                path_centers_pixels[i],
                path_centers_pixels[i + 1],
                (0, 0, 255),
                2,
            )

    # Show the updated frame
    cv2.imshow("Live Path Update", overlay_image)
    return
        





def init(frame, start):
    """
    Initialize the system, select a goal, compute the shortest path, and visualize the result.
    """
    #start = tuple(element / 10 for element in start)
    global points  # Ensure points can be accessed and modified
    points = []  # Clear any previous points

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get grid dimensions from the user
    try:
        height_division = int(input("Enter the number of rows (N): "))
        width_division = int(input("Enter the number of columns (M): "))
    except ValueError:
        print("Invalid input! Please enter integers.")
        return

    # Create costmap and compute scaling factors
    costmap, block_height, block_width = create_costmap(frame_gray, height_division, width_division)
    image_height, image_width = frame_gray.shape
    cm_per_pixel_height = 68.5 / image_height  # 59.5 cm is the real-world height
    cm_per_pixel_width = 159.5 / image_width    # 84.1 cm is the real-world width
    cm_per_pixel = (cm_per_pixel_height + cm_per_pixel_width) / 2

    mm_per_pixel_height = cm_per_pixel_height*10  
    mm_per_pixel_width = cm_per_pixel_width*10   
    mm_per_pixel = cm_per_pixel*10

    # Select the goal point
    display_image = frame.copy()
    cv2.imshow("Select Goal", display_image)
    cv2.setMouseCallback("Select Goal", select_points, display_image)
    print("Click to select the goal point.")

    # Wait until the user selects the goal
    while len(points) < 1:
        cv2.waitKey(1)  # Allow time for mouse clicks

    # Close the "Select Goal" window
    cv2.destroyWindow("Select Goal")  # Moved here after goal selection

    if len(points) < 1:
        print("No goal point selected!")
        return

    # Convert the goal point to grid coordinates
    goal = (
        points[0][0] * height_division // frame_gray.shape[0],
        points[0][1] * width_division // frame_gray.shape[1],
    )

    # Convert the start position to grid coordinates
    start_grid = (

        start[1] *frame.shape[0]/800* height_division // frame_gray.shape[0],  # y-coordinate
        start[0] * frame.shape[1]/1500* width_division // frame_gray.shape[1],  # x-coordinate
    )
    print(f"real and grid start: {start} {start_grid}")


    # Compute the shortest path using A*
    path = astar(costmap, start_grid, goal)

    if not path:
        print("No path found!")
        return
    P = np.array(path)

    # Compute differences between consecutive points
    Pp = np.diff(P, axis=0)

    # Compute delta (differences of Pp)
    delta = np.diff(Pp, axis=0)

    # Identify indices where delta equals 0 (indicating collinearity)
    indices_to_remove = np.where((delta == 0).all(axis=1))[0] + 1

    # Remove the intermediate points
    P2 = np.delete(P, indices_to_remove, axis=0)
    
    path_cm = path_pix_to_cm(path, block_width, block_height, cm_per_pixel)
    path_mm = [(x * 10, y * 10) for x, y in path_cm]

    # Convert the path to real-world coordinates (in cm)
    path_cm = path_pix_to_cm(path, block_width, block_height, cm_per_pixel)

    # Visualize the path on the frame
    path_visualization(frame, path, block_width, block_height)

    print(f"path cm:{path_cm}")

    path_mm = [(x * 10, y * 10) for x, y in path_cm]

    print("Path in mm:", path_mm)
    print("Path in grid coordinates:", path)
    print("Costmap:\n", costmap)
    print(f"Block dimensions - Height: {block_height}, Width: {block_width}")
    print("Start position (real-world coordinates):", start)
    print("Goal position (grid coordinates):", goal)
    print("Display image shape:", display_image.shape)
    print(f"CM per pixel: {cm_per_pixel}")

    return path_mm, path, costmap, block_height, block_width, start, goal, display_image, cm_per_pixel

def update(costmap, block_height, block_width, start, goal, frame, cm_per_pixel, obstacles):
    """
    Update the costmap with obstacles, compute the shortest path, and return the path in cm.
    """
    # Convert start coordinates
    robot_y = start[0] // block_height
    robot_x = start[1] // block_width
    
    print (f"obstacles: {obstacles}")

    for distance_cm, angle_deg in obstacles:
        # Adjust the obstacle angle by adding the robot's angle
        if distance_cm <=0:
            continue
        global_angle_deg = angle_deg + start[2]  # start[2] is the robot's angle1
        global_angle_rad = np.radians(global_angle_deg)

        # Convert distance to pixels
        distance_pixels = distance_cm / cm_per_pixel

        # Calculate obstacle position in pixels
        obstacle_x_pixels = robot_x * block_width + distance_pixels * np.cos(global_angle_rad)
        obstacle_y_pixels = robot_y * block_height + distance_pixels * np.sin(global_angle_rad)

        # Convert to grid coordinates
        obstacle_x_grid = int(obstacle_x_pixels // block_width)
        obstacle_y_grid = int(obstacle_y_pixels // block_height)
        
        print(f"obstacle_x_grid: {obstacle_x_grid}, obstacle_y_grid: {obstacle_y_grid}")

        # Mark obstacle on the costmap if within bounds
        if 0 <= obstacle_x_grid < costmap.shape[1] and 0 <= obstacle_y_grid < costmap.shape[0]:
            plt.imsave("before_updated.png", costmap, cmap='gray')
            costmap[obstacle_y_grid, obstacle_x_grid] = 1  # Mark as obstacle
            plt.imsave("costmap_updated.png", costmap, cmap='gray')

    # Dynamically update the start position based on the robot's position
    dynamic_start = (robot_y, robot_x)

    # Calculate the shortest path using A*
    path = astar(costmap, dynamic_start, goal)

    # Convert the path to cm
    
    
    # Visualization using 'path' instead of 'path_cm'
    overlay_image = path_visualization(frame, path, block_width, block_height)
    # Return the path in cm
    print(path)
    
    P = np.array(path)

    # Compute differences between consecutive points
    Pp = np.diff(P, axis=0)

    # Compute delta (differences of Pp)
    delta = np.diff(Pp, axis=0)

    # Identify indices where delta equals 0 (indicating collinearity)
    indices_to_remove = np.where((delta == 0).all(axis=1))[0] + 1

    # Remove the intermediate points
    P2 = np.delete(P, indices_to_remove, axis=0)

    path_cm = path_pix_to_cm(path, block_width, block_height, cm_per_pixel)
    path_mm = [(x * 10, y * 10) for x, y in path_cm]
    return path_mm, costmap



def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit.")

    start = (170, 370, 0)  # Starting point (x, y, angle)
    initialized = False  # Track if the system has been initialized
    result = None  # Store the result of `init`

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        if not initialized:
            # Initialize the system with the first frame
            print("Initializing pathfinding...")
            try:
                result = init(frame, start)
                if result:
                    (
                        path_cm,
                        path,
                        costmap,
                        block_height,
                        block_width,
                        start,
                        goal,
                        display_image,
                        cm_per_pixel,
                    ) = result
                    print("Initialization complete. Path in cm:", path_cm)
                    initialized = True
            except Exception as e:
                print(f"An error occurred during initialization: {e}")
                break
        else:
            # Detect obstacles (example placeholder, replace with real logic)
            obstacles = []  # Replace with your obstacle detection function
            
            if obstacles:  # Call update only if obstacles are detected
                print("Obstacles detected, updating path...")
                try:
                    updated_path_cm = update(
                        costmap,
                        block_height,
                        block_width,
                        start,
                        goal,
                        frame,
                        cm_per_pixel,
                        obstacles,
                    )
                    print("Updated path in cm:", updated_path_cm)
                except Exception as e:
                    print(f"An error occurred during update: {e}")
            else:
                print("No obstacles detected, no update required.")

        # Display the live video feed
        #cv2.imshow("Live Video Feed", frame)

        # Check if the user wants to quit
        key = cv2.waitKey(1) & 0xFF  # Use & 0xFF for compatibility
        if key == ord('q'):  # Quit on pressing 'q'
            print("Exiting...")
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
