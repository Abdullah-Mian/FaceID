# Import necessary libraries
# - torch: PyTorch core for model loading/inference. Why? MobileFaceNet is PyTorch-native (tensors, nn.Module).
#   What? Loads scripted model, runs forward pass on CPU.
#   Speed: ~5ms/inference on i5 (JIT-optimized); no ONNX overhead.
# - torchvision.transforms: For image preprocessing. Why? Standardizes input (resize/normalize) for pretrained weights.
#   What? Compose chain: ToTensor (uint8->float tensor), Normalize (to match training mean/std).
#   Trade-off: Fixed to ArcFace norms; custom if tweaking.
# - cv2: OpenCV for webcam, detection (Haar), display. Why? Barebone detection—no Insightface dep.
#   What? CascadeClassifier for faces (pretrained XML, ~1ms/detect).
#   Speed: 100+ FPS detection; simple but less accurate than SCRFD (95% vs 99%; ok for frontal webcam).
# - numpy: For embedding ops (norm/dot). Why? Cosine sim on 128-dim vectors.
#   What? linalg.norm, dot (SIMD-optimized).
# - pickle/os/sys: DB persistence, paths, errors. Why? Same as before; lightweight.
# - requests: For PTH download. Why? Auto-fetch if missing (caches locally, avoids re-download like Buffalo).
#   What? GET to GitHub release; saves to ./models/.
#   Speed: One-time ~4MB download (<10s).
import torch
import torchvision.transforms as transforms
import cv2  # OpenCV for camera, detection, display
import numpy as np
import pickle
import os
import sys
import requests  # For downloading PTH (pip install requests if needed, but conda has it)

# Constants
# - MODEL_URL: GitHub release link for scripted PT (TorchScript: compiled, no class needed).
#   Why? Self-contained; loads with jit.load (faster startup).
# - MODEL_PATH: Local cache dir/file. Why? Persists like ~/.insightface; checks exists to skip download.
# - EMBED_SIZE: 128 for MobileFaceNet (vs 512 in ResNet). Why? Lightweight; cosine works same.
# - THRESHOLD: 0.4 for cosine (tuned lower than 0.6; MobileFaceNet embeddings are denser).
#   Trade-off: Lower = more false positives; test on your data.
# - CAPTURE_DELAY/FPS_CAP: Same as optimized script; for smooth loop.
MODEL_URL = 'https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet_scripted.pt'
MODEL_PATH = './models/mobilefacenet_scripted.pt'
DB_FILE = 'faces_db.pkl'
EMBED_SIZE = 128
SIMILARITY_THRESHOLD = 0.4  # Lower for 128-dim; adjust 0.3-0.5
CAPTURE_DELAY = 2000
FPS_CAP = 15

# Preprocessing transform (ArcFace standard)
# Why this? Matches training: RGB 112x112, normalized to [-1,1] (mean=0.5, std=0.5 for [0,1] tensor).
# What? Chain: Resize (bilinear), ToTensor (HWC uint8 -> CHW float [0,1]), Normalize.
# How? Applied to cropped face; outputs [1,3,112,112] tensor.
transform = transforms.Compose([
    transforms.ToPILImage(),  # BGR (cv2) -> PIL RGB
    transforms.Resize((112, 112)),  # Bilinear interp; ArcFace input size
    transforms.ToTensor(),  # To [0,1] float tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # To ~[-1,1]; hypersphere friendly
])

def download_model():
    """
    Downloads PTH if not cached. Why? Ensures offline use after first run (like Buffalo fix).
    What? Creates dir, requests GET, saves binary.
    How? os.makedirs safe (no error if exists); requests stream for large files (but 4MB fast).
    Speed: <10s first; 0s after.
    Returns: Path to PTH.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Create ./models/ if missing
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MobileFaceNet model from {MODEL_URL}...")
        response = requests.get(MODEL_URL, stream=True)  # Stream: Memory-efficient
        response.raise_for_status()  # Error if 404
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                f.write(chunk)
        print("Download complete! Cached locally.")
    else:
        print("Model already cached.")
    return MODEL_PATH

def load_model():
    """
    Loads MobileFaceNet. Why separate? Isolates torch (errors early).
    What? jit.load(scripted PT): Returns TorchScriptModule (callable like model(input)).
    How? map_location='cpu' for Windows/no-GPU; eval() for inference (no grads).
    Speed: <100ms load (JIT pre-compiled).
    Returns: model (nn.Module-like).
    """
    try:
        pth_path = download_model()
        model = torch.jit.load(pth_path, map_location='cpu')  # CPU-only; no CUDA
        model.eval()  # Inference mode: Disables dropout/batchnorm updates
        print(f"MobileFaceNet loaded! Embedding size: {EMBED_SIZE}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Check PTH download; ensure torch installed.")
        sys.exit(1)

def load_database():
    """
    Same as before: Pickle load {name: np.array(128-dim)}.
    Why? Persistence; unchanged.
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
    Same: Pickle dump.
    """
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db, f)
    print(f"Saved DB with {len(db)} faces.")

def detect_face(frame):
    """
    Simple face detection with Haar cascade. Why? Barebone—no Insightface/MTCNN (avoids deps/errors).
    What? Loads XML classifier (built-in OpenCV); detectMultiScale on grayscale.
    How? scaleFactor=1.1 (pyramid step), minNeighbors=5 (false positive filter).
    Returns: List of (x,y,w,h) bboxes or [].
    Speed: ~1ms/frame; 95% accuracy frontal; misses side profiles (trade-off for simplicity).
    """
    # Load cascade (downloads XML if missing to cv2 data dir)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale for faster detect
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces  # np.array of [[x,y,w,h], ...]

def extract_embedding(model, face_crop):
    """
    Extracts 128-dim embedding from cropped face.
    Why? Core recognition: Forward pass on preprocessed crop.
    What? Apply transform (112x112 norm), model(input_tensor), squeeze to [128].
    How? face_crop: BGR ROI from bbox; to RGB via PIL; batch=1.
    Speed: ~5ms on i5 (lightweight CNN).
    Returns: np.array(128,) or None (error).
    """
    try:
        # Crop is BGR (cv2); transform handles to RGB/tensor
        input_tensor = transform(face_crop).unsqueeze(0)  # Add batch dim: [1,3,112,112]
        with torch.no_grad():  # No grads: Saves mem/CPU
            embedding = model(input_tensor)  # Forward: TorchScript call
        embedding = embedding.squeeze().cpu().numpy()  # [128] float32 to np
        return embedding / np.linalg.norm(embedding)  # Pre-normalize for cosine (unit vector)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def capture_face(model, mode='add'):
    """
    Webcam capture + detect + extract. Why? End-to-end like before, but barebone.
    What? Loop: Read frame, detect (Haar), draw boxes, key check, on SPACE: Crop first face, extract.
    Optimizations: FPS cap, mirror flip, downsample display (640x480).
    Returns: Normalized embedding or None.
    Trade-off: Haar misses non-frontal; add MTCNN later if needed (pip facenet-pytorch).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam error.")
        return None
    cap.set(cv2.CAP_PROP_FPS, FPS_CAP)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"{mode.capitalize()}: SPACE to capture, ESC quit.")
    embedding = None
    delay = int(1000 / FPS_CAP)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Mirror
        display_frame = cv2.resize(frame, (640, 480))  # Downsample for speed
        faces = detect_face(display_frame)  # Detect on display (low-res ok for Haar)

        # Draw boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(display_frame, "SPACE capture, ESC quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Capture', display_frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if len(faces) == 0:
                print("No face!")
                continue
            (x, y, w, h) = faces[0]  # First face
            face_crop = display_frame[y:y+h, x:x+w]  # ROI crop
            print("Hold still...")
            cv2.waitKey(CAPTURE_DELAY)

            embedding = extract_embedding(model, face_crop)
            if embedding is not None:
                print(f"Captured! Shape: {embedding.shape}")
                break

    cap.release()
    cv2.destroyAllWindows()
    return embedding

def add_face(model, db):
    """
    Enroll: Name input, capture, store.
    Why? Same CLI; modifies db in-place.
    """
    name = input("Name: ").strip()
    if not name or name in db and input(f"Overwrite {name}? (y/n): ").lower() != 'y':
        return
    embedding = capture_face(model, 'add')
    if embedding is not None:
        db[name] = embedding
        save_database(db)
        print(f"Added {name}.")

def verify_face(model, db):
    """
    Verify: Capture, compare cosine to all.
    Why? Loop over small DB; max score > thresh.
    What? Pre-normalized, so dot = cosine.
    """
    if not db:
        print("Empty DB!")
        return
    embedding = capture_face(model, 'verify')
    if embedding is None:
        return

    best_score, best_name = -1, None
    for name, stored in db.items():
        score = np.dot(embedding, stored)  # Cosine (both unit)
        print(f"{name}: {score:.3f}")
        if score > best_score:
            best_score, best_name = score, name

    if best_score > SIMILARITY_THRESHOLD:
        print(f"Match: {best_name} ({best_score:.3f})")
    else:
        print(f"No match (best: {best_name}, {best_score:.3f})")

def main():
    """
    Orchestrate: Load once, CLI loop.
    Why? Reuse model/DB; auto-save on quit.
    """
    model = load_model()
    db = load_database()

    print("MobileFaceNet Ready! Commands: add, verify, quit")
    while True:
        cmd = input("\nCommand: ").strip().lower()
        if cmd == 'add':
            add_face(model, db)
        elif cmd == 'verify':
            verify_face(model, db)
        elif cmd == 'quit':
            save_database(db)
            break
        else:
            print("Invalid.")

if __name__ == "__main__":
    main()