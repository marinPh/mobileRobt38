import cv2
import numpy as np
from heapq import heappush, heappop

# GLOBAL VARIABLES ============================================
# Load the image
image_path = r"C:/Users/neilc/OneDrive/Bureau/Exercises_mobile/mobileRobt38/table2.jpeg"
print(f"Loading image from: {image_path}")
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    print("Error: Image could not be loaded. Check the file path and format.")
    exit(1)

# Global list to store user-selected points
points = []


def select_points(event, x, y, flags, param):
    """
    Mouse callback function to capture the start and goal points.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point selected at: ({x}, {y})")
        points.append((y, x))  # Append the selected point
        temp_image = param.copy()
        color = (255, 0, 0) if len(points) == 1 else (0, 0, 255)  # Blue for start, red for goal
        cv2.circle(temp_image, (x, y), 5, color, -1)  # Draw a circle for the selected point
        cv2.imshow("Select Start and Goal", temp_image)

        # Close the window if two points are selected
        if len(points) == 2:
            cv2.destroyWindow("Select Start and Goal")


def create_costmap(image, grid_rows, grid_cols):
    """
    Discretize the binary image into a costmap.
    """
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    height, width = image.shape
    block_height = height // grid_rows
    block_width = width // grid_cols
    costmap = np.zeros((grid_rows, grid_cols), dtype=np.int8)

    for i in range(grid_rows):
        for j in range(grid_cols):
            block = binary_image[i * block_height: (i + 1) * block_height,
                                 j * block_width: (j + 1) * block_width]
            if np.mean(block) > 127:  # Check for walkable region (white)
                costmap[i, j] = 0  # Walkable
            else:
                costmap[i, j] = 1  # Obstacle
    return costmap, block_height, block_width


def reconstruct_binary_image_from_costmap(costmap, block_height, block_width, image_shape):
    """
    Reconstruct a binary image from the costmap for visualization.
    """
    rows, cols = costmap.shape
    reconstructed_image = np.zeros(image_shape, dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            value = 255 if costmap[i, j] == 0 else 0  # 255 for walkable, 0 for obstacles
            reconstructed_image[
                i * block_height : (i + 1) * block_height,
                j * block_width : (j + 1) * block_width,
            ] = value

    return reconstructed_image


def heuristic(a, b):
    """
    Heuristic function for A* algorithm (Manhattan distance).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(costmap, start, goal):
    """
    A* algorithm for shortest path.
    """
    rows, cols = costmap.shape
    open_set = []
    heappush(open_set, (0, 0, start))  # (f_cost, g_cost, position)
    came_from = {}
    g_costs = {start: 0}
    explored = set()

    def neighbors(node):
        x, y = node
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and costmap[nx, ny] == 0:
                yield (nx, ny)

    while open_set:
        _, current_g_cost, current_pos = heappop(open_set)
        explored.add(current_pos)

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


def update(costmap, block_height, block_width, start, goal, display_image, obstacles, cm_per_pixel):
    """
    Update the costmap with obstacles and compute the shortest path.
    """
    path = astar(costmap, start, goal)

    # Draw path on the original image
    if path:
        overlay_image = display_image.copy()
        path_centers = []

        for row, col in path:
            # Calculate the center of the square
            center_x = int((col + 0.5) * block_width)
            center_y = int((row + 0.5) * block_height)
            path_centers.append((center_x, center_y))
            # Draw the center point
            cv2.circle(overlay_image, (center_x, center_y), 5, (0, 255, 0), -1)

        # Draw lines connecting the centers
        for i in range(len(path_centers) - 1):
            cv2.line(
                overlay_image, path_centers[i], path_centers[i + 1], (0, 0, 255), 2
            )

        cv2.imshow("Shortest Path", overlay_image)

    return path, costmap


def main():
    """
    Main function to initialize, select start and goal, and compute the path.
    """
    global points

    # Ask for grid dimensions
    try:
        height_division = int(input("Enter the number of rows (N): "))
        width_division = int(input("Enter the number of columns (M): "))
    except ValueError:
        print("Invalid input! Please enter integers.")
        return

    # Display the original image for both start and goal selection
    display_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Select Start and Goal", display_image)

    # Select start and goal points
    cv2.setMouseCallback("Select Start and Goal", select_points, display_image)
    print("Click to select the start (blue) and goal (red) points.")

    # Wait for the points to be selected and the window to close automatically
    while len(points) < 2:
        cv2.waitKey(1)

    start_pixel, goal_pixel = points[0], points[1]

    # Create the costmap
    costmap, block_height, block_width = create_costmap(original_image, height_division, width_division)

    # Calculate real-world scaling
    cm_per_pixel_height = 40 / original_image.shape[0]
    cm_per_pixel_width = 80 / original_image.shape[1]
    cm_per_pixel = (cm_per_pixel_height + cm_per_pixel_width) / 2

    # Convert start and goal to grid coordinates
    start = (
        start_pixel[0] * height_division // original_image.shape[0],
        start_pixel[1] * width_division // original_image.shape[1],
    )
    goal = (
        goal_pixel[0] * height_division // original_image.shape[0],
        goal_pixel[1] * width_division // original_image.shape[1],
    )

    # Example obstacles (optional)
    obstacles = []

    # Compute and visualize the path
    path, updated_costmap = update(costmap, block_height, block_width, start, goal, display_image, obstacles, cm_per_pixel)

    if path:
        print("Path found (in grid coordinates):", path)
    else:
        print("No path found!")

    # Reconstruct and display the binary image from the costmap
    reconstructed_image = reconstruct_binary_image_from_costmap(updated_costmap, block_height, block_width, original_image.shape)
    cv2.imshow("Reconstructed Binary Image from Costmap", reconstructed_image)

    # Wait for 'q' to close all windows
    print("Press 'q' to close all windows.")
    while True:
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
