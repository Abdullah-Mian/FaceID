import cv2
import time
from collections import deque
import numpy as np

# Install required package:
# pip install opencv-python

print("Initializing face detector...")

# Use OpenCV's DNN face detector (ResNet-based, very accurate and fast)
# Download the model files if needed
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"

try:
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    print("Model loaded successfully!")
except:
    print("\nModel files not found. Downloading...")
    print("Please download these two files to the script directory:")
    print("1. https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
    print("2. https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
    print("\nOr use this automatic download:")
    
    import urllib.request
    
    # Download model files
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    
    print("Downloading model file...")
    urllib.request.urlretrieve(model_url, modelFile)
    print("Downloading config file...")
    urllib.request.urlretrieve(config_url, configFile)
    print("Download complete!")
    
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS calculation variables
fps_queue = deque(maxlen=30)  # Rolling average over 30 frames
prev_time = time.time()
frame_count = 0

print("\nCamera started!")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Start timing for FPS calculation
    current_time = time.time()
    frame_count += 1
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Prepare blob for DNN input
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        1.0, 
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    
    # Pass blob through network
    net.setInput(blob)
    detections = net.forward()
    
    # Process detections
    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.5:
            face_count += 1
            
            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display confidence
            label = f"Face: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Calculate accurate FPS
    time_diff = current_time - prev_time
    if time_diff > 0:
        fps = 1.0 / time_diff
        fps_queue.append(fps)
    
    # Get average FPS
    if len(fps_queue) > 0:
        avg_fps = sum(fps_queue) / len(fps_queue)
    else:
        avg_fps = 0
    
    prev_time = current_time
    
    # Display FPS on frame (yellow text with black outline for visibility)
    fps_text = f"FPS: {avg_fps:.2f}"
    cv2.putText(frame, fps_text, (12, 32), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame, fps_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Display face count
    count_text = f"Faces: {face_count}"
    cv2.putText(frame, count_text, (12, 72), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, count_text, (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Display the frame
    cv2.imshow('Face Detection - Press Q to Quit', frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

if len(fps_queue) > 0:
    print(f"\nFinal Average FPS: {sum(fps_queue) / len(fps_queue):.2f}")
print(f"Total frames processed: {frame_count}")