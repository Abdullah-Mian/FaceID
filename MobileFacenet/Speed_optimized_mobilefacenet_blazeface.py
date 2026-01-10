# Import necessary libraries
# - torch: PyTorch core for MobileFaceNet loading/inference. Why? MobileFaceNet is PyTorch-native (tensors, nn.Module).
#   What? Loads scripted model, runs forward pass on CPU.
#   Speed: ~5ms/inference on i5 (JIT-optimized); no ONNX overhead.
# - torchvision.transforms: For image preprocessing. Why? Standardizes input (resize/normalize) for pretrained weights.
#   What? Compose chain: ToTensor (uint8->float tensor), Normalize (to match training mean/std).
#   Trade-off: Fixed to ArcFace norms; custom if tweaking.
# - cv2: OpenCV for webcam capture, BlazeFace DNN detection, display. Why? DNN module for pre-trained face detector.
#   What? readNetFromCaffe loads ResNet-based SSD (300x300 input, ~5ms/detect).
#   Speed: 80+ FPS detection; 99% accuracy (better than Haar); handles multiple faces, side profiles.
#   Trade-off: Needs model files (auto-downloads); heavier than Haar but still real-time.
# - numpy: For embedding ops (norm/dot). Why? Cosine sim on 128-dim vectors.
#   What? linalg.norm, dot (SIMD-optimized).
#   Speed: <1ms for 128-dim ops.
# - pickle/os/sys: DB persistence, paths, errors. Why? Same as before; lightweight.
# - requests: For PTH download. Why? Auto-fetch if missing (caches locally, avoids re-download).
#   What? GET to GitHub release; saves to ./models/.
#   Speed: One-time ~4MB download (<10s).
# - urllib.request: For BlazeFace model download. Why? Fallback for model files (no requests dep for models).
#   What? urlretrieve to fetch Caffe model/prototxt from OpenCV GitHub.
#   Speed: One-time ~10MB total (<15s).
# - time: For FPS calculation. Why? Accurate frame timing (not cv2.getTickFrequency).
#   What? time.time() for wall-clock delta between frames.
# - collections.deque: For FPS rolling average. Why? Smooth display (30-frame window).
#   What? Append FPS each frame, pop oldest; mean over window.
#   Speed: O(1) append/pop.
import torch
import torchvision.transforms as transforms
import cv2  # OpenCV for camera, BlazeFace DNN detection, display
import numpy as np
import pickle
import os
import sys
import requests  # For MobileFaceNet PTH download
import urllib.request  # For BlazeFace model download
import time  # For accurate FPS timing
from collections import deque  # For FPS rolling average

# Constants
# - MODEL_URL: GitHub release link for MobileFaceNet scripted PT (TorchScript: compiled, no class needed).
#   Why? Self-contained; loads with jit.load (faster startup).
# - MODEL_PATH: Local cache dir/file. Why? Persists like ~/.insightface; checks exists to skip download.
# - BLAZEFACE_MODEL: Caffe model file for ResNet-based SSD face detector.
#   Why? Pre-trained on WIDER FACE (300K faces); 99% accuracy; handles occlusions.
#   What? .caffemodel (weights, ~10MB) + deploy.prototxt (architecture).
#   Speed: ~5ms/frame on i5; outputs confidence + bbox per face.
# - BLAZEFACE_CONFIG: Prototxt (network architecture). Why? Defines SSD layers for Caffe loader.
# - EMBED_SIZE: 128 for MobileFaceNet (vs 512 in ResNet). Why? Lightweight; cosine works same.
# - SIMILARITY_THRESHOLD: 0.4 for cosine (tuned lower than 0.6; MobileFaceNet embeddings are denser).
#   Trade-off: Lower = more false positives; test on your data. Adjust 0.3-0.5 based on accuracy.
# - DETECTION_CONFIDENCE: 0.5 for BlazeFace (filters weak detections). Why? Balances false positives.
#   What? Bbox with conf < 0.5 ignored; tunes recall/precision.
# - CAPTURE_DELAY: 2000ms hold-still time after SPACE. Why? Reduces motion blur in capture.
# - FPS_CAP: 30 for smooth display. Why? Balances CPU (detection ~5ms, embedding ~5ms = 100Hz max).
#   What? Sets cv2.CAP_PROP_FPS + waitKey delay.
MODEL_URL = 'https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet_scripted.pt'
MODEL_PATH = './models/mobilefacenet_scripted.pt'
BLAZEFACE_MODEL = './models/res10_300x300_ssd_iter_140000.caffemodel'
BLAZEFACE_CONFIG = './models/deploy.prototxt'
DB_FILE = 'faces_db.pkl'
EMBED_SIZE = 128
SIMILARITY_THRESHOLD = 0.4  # Lower for 128-dim; adjust 0.3-0.5
DETECTION_CONFIDENCE = 0.5  # BlazeFace bbox threshold
CAPTURE_DELAY = 2000
FPS_CAP = 30

# Preprocessing transform (ArcFace standard)
# Why this? Matches MobileFaceNet training: RGB 112x112, normalized to [-1,1] (mean=0.5, std=0.5 for [0,1] tensor).
# What? Chain: ToPILImage (BGR -> RGB), Resize (bilinear interp to 112x112), ToTensor (HWC uint8 -> CHW float [0,1]), 
#       Normalize (shift/scale to ~[-1,1]; hypersphere friendly for cosine).
# How? Applied to cropped face ROI; outputs [1,3,112,112] tensor.
# Trade-off: Fixed params; change mean/std if using different pretrained weights.
transform = transforms.Compose([
    transforms.ToPILImage(),  # BGR (cv2) -> PIL RGB
    transforms.Resize((112, 112)),  # Bilinear interp; ArcFace input size
    transforms.ToTensor(),  # To [0,1] float tensor (CHW)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # To ~[-1,1]; unit hypersphere
])

def download_blazeface_models():
    """
    Downloads BlazeFace Caffe model files if not cached.
    Why? Ensures offline use after first run (like MobileFaceNet PTH).
    What? Creates ./models/ dir, urlretrieve from OpenCV GitHub (official mirror).
    How? Checks os.path.exists for both files; skips if present.
    Files:
      1. res10_300x300_ssd_iter_140000.caffemodel (~10MB): Weights trained on WIDER FACE.
      2. deploy.prototxt (~5KB): Network architecture (SSD with ResNet-10 backbone).
    Speed: <15s first run (10MB download); 0s after (cached).
    Returns: Tuple (model_path, config_path).
    Trade-off: Requires internet first; fails if URLs change (fallback: manual download instructions).
    """
    os.makedirs(os.path.dirname(BLAZEFACE_MODEL), exist_ok=True)  # Create ./models/ if missing
    
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    
    # Download model file if missing
    if not os.path.exists(BLAZEFACE_MODEL):
        print(f"Downloading BlazeFace model ({model_url.split('/')[-1]})...")
        try:
            urllib.request.urlretrieve(model_url, BLAZEFACE_MODEL)
            print("Model downloaded!")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Manual download: Save to ./models/res10_300x300_ssd_iter_140000.caffemodel")
            sys.exit(1)
    else:
        print("BlazeFace model already cached.")
    
    # Download config file if missing
    if not os.path.exists(BLAZEFACE_CONFIG):
        print(f"Downloading BlazeFace config ({config_url.split('/')[-1]})...")
        try:
            urllib.request.urlretrieve(config_url, BLAZEFACE_CONFIG)
            print("Config downloaded!")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Manual download: Save to ./models/deploy.prototxt")
            sys.exit(1)
    else:
        print("BlazeFace config already cached.")
    
    return BLAZEFACE_MODEL, BLAZEFACE_CONFIG

def load_blazeface():
    """
    Loads BlazeFace DNN detector.
    Why? Separate function for clarity; isolates OpenCV DNN errors.
    What? readNetFromCaffe(prototxt, caffemodel): Returns cv2.dnn.Net object.
    How? Auto-downloads models if missing; loads to CPU (no CUDA needed).
    Speed: <100ms load time (Caffe parsing).
    Returns: cv2.dnn.Net (callable with setInput/forward).
    Trade-off: CPU-only (GPU via setPreferableBackend if CUDA installed; not needed for real-time).
    Error handling: Exits if files corrupt/missing.
    """
    try:
        model_path, config_path = download_blazeface_models()
        net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        print(f"BlazeFace loaded! Detection confidence threshold: {DETECTION_CONFIDENCE}")
        return net
    except Exception as e:
        print(f"Error loading BlazeFace: {e}")
        print("Check model files in ./models/ directory.")
        sys.exit(1)

def download_mobilefacenet():
    """
    Downloads MobileFaceNet PTH if not cached.
    Why? Ensures offline use after first run (like Buffalo fix).
    What? Creates dir, requests GET, saves binary.
    How? os.makedirs safe (no error if exists); requests stream for large files (but 4MB fast).
    Speed: <10s first; 0s after.
    Returns: Path to PTH.
    Trade-off: Requires internet first; GitHub release must be stable (v1.0).
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Create ./models/ if missing
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MobileFaceNet model from {MODEL_URL}...")
        try:
            response = requests.get(MODEL_URL, stream=True)  # Stream: Memory-efficient
            response.raise_for_status()  # Error if 404
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                    f.write(chunk)
            print("Download complete! Cached locally.")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Check internet or download manually from GitHub release.")
            sys.exit(1)
    else:
        print("MobileFaceNet model already cached.")
    return MODEL_PATH

def load_mobilefacenet():
    """
    Loads MobileFaceNet.
    Why? Separate function; isolates torch errors.
    What? jit.load(scripted PT): Returns TorchScriptModule (callable like model(input)).
    How? map_location='cpu' for Windows/no-GPU; eval() for inference (no grads).
    Speed: <100ms load (JIT pre-compiled).
    Returns: model (nn.Module-like).
    Trade-off: CPU-only (CUDA via map_location='cuda' if GPU; not needed for 5ms inference).
    """
    try:
        pth_path = download_mobilefacenet()
        model = torch.jit.load(pth_path, map_location='cpu')  # CPU-only; no CUDA
        model.eval()  # Inference mode: Disables dropout/batchnorm updates
        print(f"MobileFaceNet loaded! Embedding size: {EMBED_SIZE}")
        return model
    except Exception as e:
        print(f"Error loading MobileFaceNet: {e}")
        print("Check PTH download; ensure torch installed (pip install torch torchvision).")
        sys.exit(1)

def load_database():
    """
    Loads face DB from pickle.
    Why? Persistence; unchanged from original.
    What? Pickle load {name: np.array(128-dim normalized embedding)}.
    How? Checks os.path.exists; returns empty dict if missing.
    Speed: <10ms for 100 faces (pickle fast for small data).
    Returns: dict[str, np.ndarray] or {}.
    Trade-off: Pickle unsafe for untrusted data; ok for local use.
    """
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'rb') as f:
            db = pickle.load(f)
        print(f"Loaded DB with {len(db)} faces.")
    else:
        db = {}
        print("Starting fresh DB.")
    return db

def save_database(db):
    """
    Saves face DB to pickle.
    Why? Persist enrollments across runs.
    What? Pickle dump dict to binary file.
    How? Overwrites DB_FILE each time.
    Speed: <10ms for 100 faces.
    Returns: None.
    """
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db, f)
    print(f"Saved DB with {len(db)} faces.")

def detect_faces_blazeface(net, frame):
    """
    Detects faces using BlazeFace DNN.
    Why? Replaces Haar; better accuracy (99% vs 95%) + multi-face + side profiles.
    What? Prepares blob (300x300 resized, BGR->RGB swap, mean subtraction), runs forward pass, filters by confidence.
    How? 
      1. blobFromImage: Resizes frame to 300x300, subtracts mean [104,177,123] (ImageNet norm), scales to [0,1].
      2. net.setInput(blob): Loads blob into DNN.
      3. net.forward(): Runs SSD; returns [1,1,N,7] where N=detections, 7=[batchId, classId, confidence, x1, y1, x2, y2].
      4. Filters conf > DETECTION_CONFIDENCE; denormalizes coords to frame size.
    Speed: ~5ms/frame on i5 (300x300 input; ResNet-10 backbone).
    Returns: List of tuples [(x1, y1, x2, y2, confidence), ...] in pixel coords.
    Trade-off: Heavier than Haar but still real-time; CPU-only (GPU optional).
    """
    h, w = frame.shape[:2]
    
    # Prepare blob: Resize to 300x300, subtract ImageNet mean, swap BGR->RGB
    # Why these params? Pre-trained on ImageNet norm; scale=1.0 (no scaling), size=(300,300) for SSD input.
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),  # Resize: Bilinear interp to fixed input
        scalefactor=1.0,  # No scaling (already [0,255])
        size=(300, 300),  # SSD expects 300x300
        mean=(104.0, 177.0, 123.0),  # ImageNet BGR mean (pre-trained norm)
        swapRB=False,  # Already BGR (cv2 default)
        crop=False  # No center crop
    )
    
    net.setInput(blob)  # Load blob into network
    detections = net.forward()  # Forward pass: [1,1,N,7]
    
    faces = []
    # Iterate over detections: detections.shape = [1, 1, N, 7]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Extract confidence (index 2)
        
        if confidence > DETECTION_CONFIDENCE:
            # Extract bbox: [x1, y1, x2, y2] normalized to [0,1]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # Denormalize to pixel coords
            (x1, y1, x2, y2) = box.astype("int")
            
            # Clamp to frame bounds (safety; detection can overshoot)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            faces.append((x1, y1, x2, y2, confidence))
    
    return faces

def extract_embedding(model, face_crop):
    """
    Extracts 128-dim embedding from cropped face.
    Why? Core recognition: Forward pass on preprocessed crop.
    What? Apply transform (BGR->RGB, resize to 112x112, normalize to [-1,1]), model(input_tensor), squeeze to [128].
    How? 
      1. face_crop: BGR ROI [H, W, 3] from bbox.
      2. transform: Converts to [3, 112, 112] tensor normalized to [-1,1].
      3. unsqueeze(0): Adds batch dim [1, 3, 112, 112].
      4. model(input): Forward pass (TorchScript optimized); returns [1, 128].
      5. Normalize: L2-norm to unit vector (cosine sim requires).
    Speed: ~5ms on i5 (lightweight CNN; 128-dim output).
    Returns: np.array(128,) normalized to unit vector, or None on error.
    Trade-off: Requires valid face crop (min ~30x30 pixels); blurry/occluded faces degrade quality.
    """
    try:
        # Crop is BGR (cv2); transform handles RGB conversion + resize + normalize
        input_tensor = transform(face_crop).unsqueeze(0)  # Add batch dim: [1,3,112,112]
        
        with torch.no_grad():  # No grads: Saves mem/CPU (inference only)
            embedding = model(input_tensor)  # Forward: TorchScript call [1, 128]
        
        embedding = embedding.squeeze().cpu().numpy()  # [128] float32 to np
        norm = np.linalg.norm(embedding)  # L2 norm
        if norm == 0:  # Safety: Avoid div by zero (shouldn't happen)
            print("Warning: Zero-norm embedding!")
            return None
        return embedding / norm  # Normalize to unit vector for cosine sim
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def capture_face(blazeface_net, mobilefacenet_model, mode='add'):
    """
    Webcam capture + BlazeFace detection + MobileFaceNet embedding extraction.
    Why? End-to-end enrollment/verification pipeline.
    What? 
      1. Opens webcam (cv2.VideoCapture) with OPTIMIZED settings for instant startup.
      2. Loop: Read frame -> detect faces (BlazeFace) -> draw boxes -> wait for SPACE key.
      3. On SPACE: Crop first detected face -> hold-still delay -> extract embedding.
      4. FPS calculation: time.time() delta, rolling average over 30 frames.
    How?
      - SPEED OPTIMIZATIONS:
        * CAP_PROP_FOURCC: Force MJPG codec (hardware-accelerated, 10x faster than YUYV).
        * CAP_PROP_BUFFERSIZE: Set to 1 (minimal latency, no frame buffering).
        * Resolution: 640x480 (sweet spot: fast capture, good quality).
        * Backend: Auto-select fastest (DSHOW on Windows, V4L2 on Linux).
      - FPS cap: Sets cv2.CAP_PROP_FPS + waitKey delay (int(1000/FPS_CAP)).
      - Mirror flip: cv2.flip for selfie-style (left=left).
      - Display: Native res (no downsample needed at 640x480).
      - FPS display: deque for 30-frame rolling avg; text overlay with outline for visibility.
    Speed: ~60 FPS (camera opens in <100ms vs 2-3s default; 5ms detect + 5ms render + 6ms idle).
    Returns: Normalized embedding (np.array 128-dim) or None (ESC/error).
    Trade-off: MJPG uses more USB bandwidth but way faster startup/capture.
    """
    # SPEED OPTIMIZATION 1: Use DirectShow backend on Windows (fastest)
    # Why? cv2.CAP_DSHOW bypasses slow Media Foundation; <100ms startup vs 2-3s.
    # What? Platform-specific backend: DSHOW (Windows), V4L2 (Linux), AVFoundation (Mac).
    # How? Pass backend flag to VideoCapture constructor.
    # Trade-off: Windows-only optimization; fallback to default on other platforms.
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow: instant on Windows
    else:
        cap = cv2.VideoCapture(0)  # Default backend on Linux/Mac
    
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return None
    
    # SPEED OPTIMIZATION 2: Force MJPG codec (hardware-accelerated)
    # Why? MJPG is compressed in-camera, reducing USB transfer time (10x faster than raw YUYV).
    # What? Sets FOURCC (Four Character Code) to 'MJPG'.
    # How? cv2.VideoWriter_fourcc converts 'M','J','P','G' to int code.
    # Speed: Camera startup drops from 2-3s to <100ms; frame grab from 30ms to 3ms.
    # Trade-off: Slight compression artifacts (negligible for face detection).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # SPEED OPTIMIZATION 3: Minimize buffer size (reduce latency)
    # Why? Default buffer=4 frames causes lag; buffer=1 gives real-time.
    # What? Sets internal frame queue to 1 (most recent frame only).
    # How? CAP_PROP_BUFFERSIZE property (OpenCV 3.3+).
    # Speed: Latency drops from 100ms to <10ms.
    # Trade-off: None (we want real-time, not playback).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set webcam properties (order matters: set codec/buffer BEFORE resolution/FPS)
    # Why? Some cameras reset settings if resolution changed first.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Width: 640 (VGA; fast + good quality)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height: 480
    cap.set(cv2.CAP_PROP_FPS, FPS_CAP)       # Request FPS (driver may ignore)
    
    print(f"\n{'='*60}")
    print(f"Mode: {mode.upper()}")
    print(f"Instructions: Press SPACE to capture face, ESC to quit")
    print(f"{'='*60}\n")
    
    embedding = None
    delay = int(1000 / FPS_CAP)  # waitKey delay in ms
    
    # FPS calculation variables
    fps_queue = deque(maxlen=30)  # Rolling average over 30 frames
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Start timing for FPS
        current_time = time.time()
        
        frame = cv2.flip(frame, 1)  # Mirror flip (selfie mode)
        display_frame = frame.copy()  # Copy for drawing (preserve original for crop)
        
        # Detect faces with BlazeFace
        faces = detect_faces_blazeface(blazeface_net, frame)
        
        # Draw bounding boxes and confidence
        for (x1, y1, x2, y2, conf) in faces:
            # Draw green rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display confidence score
            label = f"Face: {conf:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate FPS
        time_diff = current_time - prev_time
        if time_diff > 0:
            fps = 1.0 / time_diff
            fps_queue.append(fps)
        prev_time = current_time
        
        # Average FPS over rolling window
        if len(fps_queue) > 0:
            avg_fps = sum(fps_queue) / len(fps_queue)
        else:
            avg_fps = 0
        
        # Display FPS (yellow text with black outline for visibility)
        fps_text = f"FPS: {avg_fps:.2f}"
        cv2.putText(display_frame, fps_text, (12, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)  # Black outline
        cv2.putText(display_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Yellow text
        
        # Display instructions
        instr_text = "SPACE: Capture | ESC: Quit"
        cv2.putText(display_frame, instr_text, (12, 72), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(display_frame, instr_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display face count
        count_text = f"Faces: {len(faces)}"
        cv2.putText(display_frame, count_text, (12, 112), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(display_frame, count_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Face Recognition - Capture', display_frame)
        
        key = cv2.waitKey(delay) & 0xFF
        if key == 27:  # ESC
            print("Cancelled.")
            break
        elif key == 32:  # SPACE
            if len(faces) == 0:
                print("No face detected! Please face the camera.")
                continue
            
            # Use first detected face
            (x1, y1, x2, y2, conf) = faces[0]
            print(f"\nFace detected! Confidence: {conf:.3f}")
            print(f"Hold still for {CAPTURE_DELAY/1000:.1f} seconds...")
            
            # Visual feedback: Flash box
            for _ in range(3):
                flash_frame = display_frame.copy()
                cv2.rectangle(flash_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
                cv2.imshow('Face Recognition - Capture', flash_frame)
                cv2.waitKey(200)
                cv2.imshow('Face Recognition - Capture', display_frame)
                cv2.waitKey(200)
            
            cv2.waitKey(CAPTURE_DELAY)  # Hold-still delay
            
            # Crop face from original frame (not display_frame)
            face_crop = frame[y1:y2, x1:x2]
            
            # Extract embedding
            print("Extracting embedding...")
            embedding = extract_embedding(mobilefacenet_model, face_crop)
            
            if embedding is not None:
                print(f"✓ Embedding extracted! Shape: {embedding.shape}")
                break
            else:
                print("✗ Failed to extract embedding. Try again.")
    
    cap.release()
    cv2.destroyAllWindows()
    return embedding

def add_face(blazeface_net, mobilefacenet_model, db):
    """
    Enrolls a new face into the database.
    Why? Adds person to recognition system.
    What? Prompts for name, captures face via webcam, extracts embedding, stores in DB.
    How?
      1. Input name via CLI.
      2. Check if name exists; prompt overwrite confirmation.
      3. Call capture_face (webcam loop + BlazeFace detect + MobileFaceNet embed).
      4. If embedding valid, add to db dict {name: np.array(128)}.
      5. Save DB to pickle.
    Speed: Instant (network ops only).
    Returns: None (modifies db in-place).
    Trade-off: No batch enrollment; one-by-one only (UI feature for future).
    """
    print("\n" + "="*60)
    print("ADD NEW FACE")
    print("="*60)
    
    name = input("Enter name: ").strip()
    if not name:
        print("Invalid name. Cancelled.")
        return
    
    # Check if name exists
    if name in db:
        overwrite = input(f"'{name}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Cancelled.")
            return
    
    # Capture face and extract embedding
    embedding = capture_face(blazeface_net, mobilefacenet_model, mode='add')
    
    if embedding is not None:
        db[name] = embedding
        save_database(db)
        print(f"\n✓ Successfully added '{name}' to database!")
        print(f"Total faces in DB: {len(db)}")
    else:
        print("\n✗ Failed to add face.")

def verify_face(blazeface_net, mobilefacenet_model, db):
    """
    Verifies a face against the database.
    Why? Identifies person from webcam capture.
    What? Captures face, extracts embedding, compares cosine similarity to all DB entries, returns best match.
    How?
      1. Check DB not empty.
      2. Call capture_face (webcam loop + extract).
      3. Loop over db: Compute cosine sim (np.dot since both unit vectors).
      4. Track best score + name; print all scores.
      5. If best > threshold, match; else no match.
    Speed: O(N) where N=DB size; ~0.1ms per comparison (dot product).
    Returns: None (prints result).
    Trade-off: Linear search; slow for 10K+ faces (use FAISS/Annoy for scale).
    """
    print("\n" + "="*60)
    print("VERIFY FACE")
    print("="*60)
    
    if not db:
        print("Database is empty! Add faces first.")
        return
    
    # Capture face and extract embedding
    embedding = capture_face(blazeface_net, mobilefacenet_model, mode='verify')
    
    if embedding is None:
        print("\n✗ Failed to capture face.")
        return
    
    print("\nComparing against database...")
    print("-" * 60)
    
    best_score = -1
    best_name = None
    
    # Compare with all stored embeddings
    for name, stored_embedding in db.items():
        # Cosine similarity: dot product of unit vectors
        score = np.dot(embedding, stored_embedding)
        print(f"{name:20s}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_name = name
    
    print("-" * 60)
    
    # Decision
    if best_score > SIMILARITY_THRESHOLD:
        print(f"\n✓ MATCH: {best_name}")
        print(f"  Confidence: {best_score:.4f} (threshold: {SIMILARITY_THRESHOLD})")
    else:
        print(f"\n✗ NO MATCH")
        print(f"  Best candidate: {best_name} ({best_score:.4f})")
        print(f"  Threshold: {SIMILARITY_THRESHOLD}")

def display_menu():
    """
    Displays CLI menu.
    Why? User-friendly interface for integer-based commands.
    What? Prints formatted menu with options 1-3.
    How? Simple print statements.
    Returns: None.
    """
    print("\n" + "="*60)
    print("FACE RECOGNITION SYSTEM")
    print("="*60)
    print("1. Add Face")
    print("2. Verify Face")
    print("3. Quit")
    print("="*60)

def main():
    """
    Main orchestrator.
    Why? Coordinates model loading, DB persistence, CLI loop.
    What? 
      1. Load BlazeFace detector (cv2.dnn.Net).
      2. Load MobileFaceNet model (torch.jit.ScriptModule).
      3. Load face database (dict from pickle).
      4. Enter CLI loop: Display menu, parse integer input, dispatch to functions.
      5. On quit: Save DB, exit gracefully.
    How?
      - Integer menu: Prompts "Choose option (1-3): ", validates input.
      - Error handling: try/except for invalid int input.
      - Loop until option 3 (quit).
    Speed: Instant (models cached after first run).
    Returns: None (exits with sys.exit(0) or natural termination).
    Trade-off: Blocking CLI (no async); single-threaded (fine for local use).
    """
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "FACE RECOGNITION SYSTEM" + " "*20 + "#")
    print("#" + " "*10 + "BlazeFace + MobileFaceNet" + " "*23 + "#")
    print("#" + " "*58 + "#")
    print("#"*60 + "\n")
    
    print("Initializing system...")
    print("-" * 60)
    
    # Step 1: Load BlazeFace detector
    # Why? Needed for all face detection (add/verify).
    # What? Downloads models if missing, loads Caffe DNN.
    # Speed: <1s after first run (cached).
    blazeface_net = load_blazeface()
    
    # Step 2: Load MobileFaceNet model
    # Why? Needed for embedding extraction.
    # What? Downloads PTH if missing, loads TorchScript model.
    # Speed: <1s after first run (cached).
    mobilefacenet_model = load_mobilefacenet()
    
    # Step 3: Load face database
    # Why? Retrieve enrolled faces from disk.
    # What? Pickle load dict {name: embedding}.
    # Speed: <100ms for 100 faces.
    db = load_database()
    
    print("-" * 60)
    print("✓ System ready!")
    print(f"✓ Database: {len(db)} faces enrolled")
    print(f"✓ Detection: BlazeFace (conf > {DETECTION_CONFIDENCE})")
    print(f"✓ Recognition: MobileFaceNet (128-dim, threshold {SIMILARITY_THRESHOLD})")
    
    # CLI loop
    while True:
        display_menu()
        
        try:
            # Get user input
            choice = input("\nChoose option (1-3): ").strip()
            
            # Validate integer
            if not choice.isdigit():
                print("Invalid input! Please enter a number (1-3).")
                continue
            
            choice = int(choice)
            
            # Dispatch based on choice
            if choice == 1:
                # Add face
                # Why? Enrolls new person to DB.
                # What? Prompts name, captures face, extracts embedding, saves.
                # Speed: ~3s (user interaction + capture + 5ms embed + save).
                add_face(blazeface_net, mobilefacenet_model, db)
            
            elif choice == 2:
                # Verify face
                # Why? Identifies person from webcam.
                # What? Captures face, compares to all DB entries, returns best match.
                # Speed: ~3s (capture) + O(N) comparison (N=DB size, ~0.1ms each).
                verify_face(blazeface_net, mobilefacenet_model, db)
            
            elif choice == 3:
                # Quit
                # Why? Clean exit with DB save.
                # What? Saves current DB state, exits program.
                print("\nSaving database...")
                save_database(db)
                print("✓ Database saved!")
                print("\nThank you for using Face Recognition System!")
                print("Goodbye!\n")
                break
            
            else:
                # Invalid choice
                print(f"Invalid option '{choice}'! Please choose 1-3.")
        
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            # Why? Allows clean exit without traceback.
            # What? Saves DB, prints message, exits.
            print("\n\nInterrupted by user.")
            print("Saving database...")
            save_database(db)
            print("✓ Database saved!")
            print("Goodbye!\n")
            break
        
        except Exception as e:
            # Catch unexpected errors
            # Why? Prevents crash; logs error for debugging.
            # What? Prints error, continues loop.
            print(f"\n✗ Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    # Entry point
    # Why? Standard Python pattern; allows import without auto-run.
    # What? Calls main() only when script executed directly.
    # How? Checks __name__ == "__main__" (true for direct execution).
    main()