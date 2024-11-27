import cv2
import threading
from queue import Queue
from matplotlib import pyplot as plt

# Thread-safe queue to hold frames
frame_queue = Queue(maxsize=5)

# Function to capture frames from the webcam
def capture_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam (ID 0 is default)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Put the frame in the queue
        if not frame_queue.full():
            frame_queue.put(frame)
        
    
    cap.release()
    

def show_frame(frame, title="Frame"):
    # Convert the frame from BGR (OpenCV format) to RGB (Matplotlib format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show(block=False)  # Display the image without blocking
    plt.pause(0.001)  # Pause briefly to allow the plot to update

# Function to process frames and apply overlays/calculations
def process_frames():
    while True:
        if not frame_queue.empty():
            # Get a frame from the queue
            frame = frame_queue.get()
            
            # Example overlay: draw a rectangle
            overlay_frame = frame.copy()
            cv2.rectangle(overlay_frame, (50, 50), (200, 200), (0, 255, 0), 2)
            
            # Example calculation: average pixel intensity
            avg_intensity = overlay_frame.mean()
            print(f"Average Intensity: {avg_intensity}")
            
            # Show the processed frame
            show_frame(frame, "Processed Frame")

            
    
"""
capture_thread = threading.Thread(target=capture_frames, daemon=True)
process_thread = threading.Thread(target=process_frames, daemon=True)

capture_thread.start()
process_thread.start()

cv2.destroyAllWindows()"""

