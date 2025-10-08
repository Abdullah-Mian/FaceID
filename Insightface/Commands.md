# InsightFace Overview

InsightFace is based in ArcFace and  
ArcFace is not a model architecture but a training loss function that enhances face embeddings (feature vectors) for better discrimination.

---

## InsightFace → the library / framework

InsightFace is the Python library (a toolkit) that handles:

- Loading models (like Buffalo)
- Detecting faces
- Aligning faces
- Generating embeddings
- Comparing embeddings (for recognition)

It’s like OpenCV, but specialized for face recognition using deep learning.

---

## Buffalo → the actual pretrained model

“Buffalo” models are the trained neural networks that InsightFace uses internally.

There are several versions:

| Model Name   | Description                | Size    | Speed   |
|--------------|----------------------------|---------|---------|
| buffalo_s    | Small, fast, lower accuracy| ~70 MB  | ⚡ Fast  |
| buffalo_m    | Medium balance             | ~150 MB | ⚙️ Medium|
| buffalo_l    | Large, high accuracy       | ~300 MB | 🐢 Slower|

---

### Code line | Model used | Comment

- `app.prepare()` → **buffalo_l** (Default, slow, high accuracy)
- `app.prepare(model='buffalo_m')` → **buffalo_m** (Balanced speed/accuracy)
- `app.prepare(model='buffalo_s')` → **buffalo_s** (Fastest, lower accuracy)

---

## Official Insight Face Repo

[https://github.com/deepinsight/insightface.git](https://github.com/deepinsight/insightface.git)

---

## Setup Commands

### Command to install Miniconda

```powershell
wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -outfile ".\miniconda.exe"
Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S" -Wait
del .\miniconda.exe
```

Add Conda to your user variables PATH manually:

- `C:\Users\Abdullah\miniconda3\Scripts`
- `C:\Users\Abdullah\miniconda3\Library\bin`

```powershell
conda init powershell

notepad $PROFILE
# >>> conda initialize >>>
& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1" | Out-String | Invoke-Expression
# <<< conda initialize <<<

conda activate base
```

---

### Create and activate environment

```powershell
conda create -n face_recog python=3.10
conda activate face_recog
```

---

`insightface: The core library for face detection, alignment, and recognition using pretrained ArcFace models (MobileFaceNet is a lightweight backbone variant, but insightface uses efficient ArcFace-based models auto-downloaded on first run; it's optimized for real-time use and very similar in architecture/purpose to pure MobileFaceNet).`

```powershell
pip install insightface==0.7.3
```

Install directly from the wheel URL (pip handles the download):

```powershell
pip install https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl
```

**OR**

Install Microsoft C++ Build Tools

---

`onnxruntime: Runtime for executing ONNX models (insightface uses ONNX format for CPU/GPU inference). On Windows, this is CPU-only by default; for GPU, you'd need CUDA setup, but we'll stick to CPU for simplicity.`

```powershell
pip install onnxruntime
```

---

`opencv-python: For camera access, image capture, and basic processing (e.g., displaying video feed).`

```powershell
pip install opencv-python
```

---

`numpy: For numerical operations on embeddings (face feature vectors).`

```powershell
pip install numpy
```

---

`pickle-mixin: For saving/loading the face database (embeddings) to/from a file. (Note: Standard pickle is built-in, but this ensures compatibility.)`

```powershell
pip install pickle-mixin
```

> DEPRECATION: Building 'pickle-mixin' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'pickle-mixin'. Discussion can be found at https://github.com/pypa/pip/issues/6334