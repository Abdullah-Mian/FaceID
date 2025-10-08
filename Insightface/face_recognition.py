# Import necessary libraries
# - insightface: Handles face detection, alignment, and embedding extraction using pretrained models.
# - cv2 (OpenCV): For accessing the webcam, capturing frames, and displaying video.
# - numpy: For mathematical operations on face embeddings (vectors).
# - pickle: For serializing (saving/loading) the face database to a file.
# - os: For file path operations (e.g., checking if DB file exists).
# - sys: For graceful exit on errors.
import insightface
import cv2  # OpenCV for camera and image handling
import numpy as np
import pickle  # Built-in for saving/loading Python objects (face DB)
import os
import sys

# Define paths and constants
# - DB_FILE: The file where we store the face database (name -> embedding vector).
# - CAPTURE_DELAY: Milliseconds to wait before/after capture for user to position face.
# - SIMILARITY_THRESHOLD: Cosine similarity threshold for verification (0.6 is a good starting point; higher = stricter match).
#   Cosine similarity measures how similar two vectors are (1.0 = identical, 0 = orthogonal).
DB_FILE = 'faces_db.pkl'  # Pickle file for storing {name: embedding} dictionary
CAPTURE_DELAY = 2000  # 2 seconds delay for user to prepare
SIMILARITY_THRESHOLD = 0.6  # Adjust based on testing; 0.5-0.7 works for good lighting

def load_face_model():
    """
    Loads the pretrained face analysis model from insightface.
    - FaceAnalysis: A wrapper that loads detection (SCRFD/RetinaFace), landmarks (2D/3D), and recognition (ArcFace) models.
    - providers=['CPUExecutionProvider']: Runs on CPU (safe for Windows without GPU setup).
    - prepare(): Downloads models if needed (e.g., buffalo_l.zip) and prepares for inference.
    Returns the app instance for use in detection/embedding.
    """
    print("Loading face recognition model... (This may take a minute on first run due to auto-download)")
    try:
        # Initialize FaceAnalysis with CPU provider (change to ['CUDAExecutionProvider'] if you have NVIDIA GPU + CUDA)
        app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        # Prepare the model: Downloads ~200-300MB models to ~/.insightface/models/ if not present (e.g., det_10g.onnx, w600k_r50.onnx).
        # det_size=(640, 640): Input size for detection (balanced speed/accuracy; larger = better but slower).
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("Model loaded successfully!")
        return app
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure insightface and onnxruntime are installed correctly.")
        sys.exit(1)

def load_database():
    """
    Loads the existing face database from pickle file.
    - If file exists, unpickles the dict {name: np.array(embedding)}.
    - If not, starts with empty dict.
    Returns the database dict.
    """
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'rb') as f:
            db = pickle.load(f)
        print(f"Loaded database with {len(db)} known faces.")
    else:
        db = {}
        print("No existing database found. Starting fresh.")
    return db

def save_database(db):
    """
    Saves the database dict to pickle file.
    - Overwrites if exists.
    - Use 'wb' mode for binary write.
    """
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db, f)
    print(f"Saved database with {len(db)} known faces.")

def capture_face(app, mode='add'):
    """
    Captures a face from webcam using OpenCV.
    - Opens camera (index 0 = default webcam).
    - Displays live feed with instructions.
    - Detects/aligns/extracts embedding using app.get() (integrated pipeline: detection + recognition in one call).
    - Handles single face assumption (takes first detected face).
    - mode: 'add' for enrollment, 'verify' for testing.
    Returns the embedding (np.array) or None if no face detected.
    """
    cap = cv2.VideoCapture(0)  # Open default camera (0). Returns True if successful.
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print(f"{mode.capitalize()}ing face: Position your face in frame and press SPACE to capture. Press ESC to cancel.")
    embedding = None

    while True:
        ret, frame = cap.read()  # Read frame: ret=True if successful, frame=BGR image (OpenCV format).
        if not ret:
            print("Error capturing frame.")
            break

        # Flip frame horizontally for mirror effect (feels natural, like a mirror).
        frame = cv2.flip(frame, 1)

        # FIXED: Use app.get(frame) for full pipeline (detection + landmarks + embedding).
        # - Returns list of dicts: [{'bbox': [x1,y1,x2,y2], 'kps': keypoints, 'det_score': confidence, 'embedding': 512-dim vector}, ...]
        # - No separate app.detect(); get() handles it efficiently.
        # - Computes everything per frame (ok for real-time; ~15 FPS on CPU).
        faces = app.get(frame)

        # Draw bounding boxes if faces detected (for visualization; uses bbox from each face dict).
        for face in faces:
            bbox = face['bbox'].astype(int)  # Convert to int for drawing.
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Green box around face.

        # Add text overlay for instructions (white text on top-left).
        cv2.putText(frame, "Press SPACE to capture, ESC to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Face Capture', frame)  # Display the frame in a window.

        key = cv2.waitKey(1) & 0xFF  # Wait for keypress (1ms delay for smooth video).

        if key == 27:  # ESC key (ASCII 27) to cancel.
            print("Capture cancelled.")
            break
        elif key == 32:  # SPACE key (ASCII 32) to capture.
            if len(faces) == 0:
                print("No face detected! Try again.")
                continue
            # Assume first (most confident) face.
            face = faces[0]
            if face['det_score'] < 0.5:  # Confidence threshold (0.5 = medium; adjust for sensitivity).
                print("Low confidence detection. Try better lighting/angle.")
                continue

            print("Capturing... Please hold still.")
            cv2.waitKey(CAPTURE_DELAY)  # Pause for positioning (optional; gives time post-press).

            # FIXED: Use pre-computed embedding from the last app.get() call (no recompute needed).
            # - face['embedding'] is the 512-dim vector (np.array) from ArcFace model.
            embedding = face['embedding']
            print(f"Face captured! Embedding shape: {embedding.shape}")
            break

    cap.release()  # Release camera resource (important to free webcam).
    cv2.destroyAllWindows()  # Close all OpenCV windows.
    return embedding

def add_face(app, db):
    """
    Adds a new face to the database.
    - Asks for name via input.
    - Captures embedding.
    - Stores in db as np.array (for easy similarity calc).
    - Saves DB.
    """
    name = input("Enter name for this face: ").strip()
    if not name:
        print("Invalid name.")
        return
    if name in db:
        overwrite = input(f"Face '{name}' exists. Overwrite? (y/n): ").lower() == 'y'
        if not overwrite:
            return

    embedding = capture_face(app, 'add')
    if embedding is not None:
        db[name] = embedding  # Store as numpy array (512-dim float32).
        save_database(db)
        print(f"Added '{name}' to database.")
    else:
        print("Failed to capture face.")

def verify_face(app, db):
    """
    Verifies a captured face against database.
    - Captures embedding.
    - Computes cosine similarity with each stored embedding.
    - Cosine sim = (A · B) / (||A|| * ||B||); higher = more similar (ranges -1 to 1).
    - Finds max similarity; if > threshold, matches name.
    - If no match or empty DB, reports unknown.
    """
    if not db:
        print("No faces in database. Add some first!")
        return

    embedding = capture_face(app, 'verify')
    if embedding is None:
        print("Failed to capture face.")
        return

    # Normalize embeddings for cosine similarity (projects to unit sphere; ignores magnitude).
    query_norm = embedding / np.linalg.norm(embedding)  # L2 (Euclidean) norm.

    best_match = None
    best_score = -1

    for name, stored_emb in db.items():
        stored_norm = stored_emb / np.linalg.norm(stored_emb)
        # Cosine similarity: dot product of normalized vectors (fast O(512) per comparison).
        score = np.dot(query_norm, stored_norm)
        print(f"Similarity with '{name}': {score:.3f}")
        if score > best_score:
            best_score = score
            best_match = name

    if best_score > SIMILARITY_THRESHOLD:
        print(f"Match found: '{best_match}' (similarity: {best_score:.3f})")
    else:
        print(f"No match found (best: '{best_match}', similarity: {best_score:.3f})")

def main():
    """
    Main function: Loads model and DB, then loops for user commands.
    - Commands: 'add' to enroll, 'verify' to check, 'quit' to exit.
    - Runs until quit.
    """
    app = load_face_model()
    db = load_database()

    print("\nFace Recognition System Ready!")
    print("Commands: 'add' (enroll face), 'verify' (check face), 'quit' (exit)")

    while True:
        cmd = input("\nEnter command: ").strip().lower()
        if cmd == 'add':
            add_face(app, db)
        elif cmd == 'verify':
            verify_face(app, db)
        elif cmd == 'quit':
            print("Goodbye!")
            break
        else:
            print("Unknown command. Try 'add', 'verify', or 'quit'.")

if __name__ == "__main__":
    main()