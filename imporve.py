import cv2
import numpy as np

cap = cv2.VideoCapture('aks.mov')

# Get original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = 1920
height = 1080

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
out = cv2.VideoWriter('output_sharp.mp4', fourcc, fps, (width, height))

# Define a sharpening kernel
sharpen_kernel = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to 1080p
    frame = cv2.resize(frame, (width, height))

    # Apply sharpening filter
    sharp = cv2.filter2D(frame, -1, sharpen_kernel)

    # Write processed frame
    out.write(sharp)

cap.release()
out.release()
print("Done! Saved as output_sharp.mp4")
