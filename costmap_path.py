#
#
#
#
#
# LIBRARIES ============================================
import cv2
import numpy as np
from heapq import heappush, heappop

# GLOBAL VARIABLES ============================================
# Load the image
image_path = r"./grid.png"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# START OF LOIC FUNCTIONS ============================================



# START OF NEIL FUNCTIONS ============================================
# Function to discretize image into a costmap
def create_costmap(image, grid_size):
    height, width = image.shape
    block_size = height // grid_size
    costmap = np.zeros((grid_size, grid_size), dtype=np.int8)

    for i in range(grid_size):
        for j in range(grid_size):
            block = image[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size]
            if np.mean(block) > 127:  # Assume white is walkable (mean > 127)
                costmap[i, j] = 0  # Walkable
            else:
                costmap[i, j] = 1  # Obstacle
    return costmap, block_size

# Heuristic function for A*
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* algorithm for shortest path
def astar(costmap, start, goal):
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

# Mouse callback to get points
points = []
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: ({x}, {y})")  # Debug print
        points.append((y, x))  # Append (row, col) in grid terms
        # Display the click on the image
        temp_image = param.copy()
        cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)  # Draw a red dot for the click
        cv2.imshow("Select Start and Goal", temp_image)


# DEF OF MAIN ============================================================


def init(original_image):
    try:
        grid_size = int(input("Enter the grid size (N): "))
    except ValueError:
        print("Invalid input! Please enter an integer.")
        return
    
    costmap, block_size = create_costmap(original_image, grid_size)

    # Show the original image for user to click points
    display_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Select Start and Goal", display_image)
    cv2.setMouseCallback("Select Start and Goal", select_points, display_image)
    print("Click two points: Start and Goal.")
    cv2.waitKey(0)

    if len(points) < 2:
        print("Please select two points!")
        return

    # Map points to grid
    start = (points[0][0] * grid_size // original_image.shape[0], 
             points[0][1] * grid_size // original_image.shape[1])
    goal = (points[1][0] * grid_size // original_image.shape[0], 
            points[1][1] * grid_size // original_image.shape[1])
    
    
    return costmap, block_size, start, goal, display_image

def update(costmap, block_size, start, goal, display_image):

    # Calculate shortest path using A*
    path = astar(costmap, start, goal)

    # Draw path on the image
    if path:
        overlay_image = display_image.copy()
        path_centers = []

        for (row, col) in path:
            # Calculate the center of the square
            center_x = int((col + 0.5) * block_size)
            center_y = int((row + 0.5) * block_size)
            path_centers.append((center_x, center_y))
            # Draw the center point
            cv2.circle(overlay_image, (center_x, center_y), 5, (0, 255, 0), -1)

        # Draw lines connecting the centers
        for i in range(len(path_centers) - 1):
            cv2.line(overlay_image, path_centers[i], path_centers[i + 1], (0, 0, 255), 2)

        cv2.imshow("Shortest Path", overlay_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return path,costmap

# LAUNCH OF MAIN ============================================================
main()



# run -> write the discretization of the N*N grid (10 will discretize the image in a grid of size 10*10) -> click on 2 points of the image then any key (usually q) and the path should appear. press q to close the window.
