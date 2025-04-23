# FaceCam-Toolkit

FaceCam-Toolkit is a collection of lightweight Python scripts that turn any webcam into a playground for computer-vision experiments.  
It demos live video preview, real-time face detection with bounding boxes/landmarks, and GDPR-friendly automatic face-blurring—all in fewer than 300 lines of code.  
Powered by **OpenCV**, **PyTorch**, and **facenet-pytorch**, the project runs on Linux, macOS, and Windows with the same set of Python-level dependencies.

---

## 1 . Quick start

```bash
# Clone your repo (if you haven’t already)
git clone <your-repo-url>
cd <your-repo>

# Create & activate a virtual environment (recommended)
python3 -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\activate
```

---

## 2 . Dependencies (pip install)

The list is identical on all three OSes; only the **torch** wheel differs (CPU vs. GPU):

| Package            | Minimum tested version | Notes                                   |
|--------------------|------------------------|-----------------------------------------|
| torch              | 2.2.1                  | Use the wheel suggested at <https://pytorch.org> for your OS / CUDA. |
| torchvision        | 0.17.1                 | Matches the torch version.              |
| facenet-pytorch    | 2.5.2                  | High-level MTCNN face detector.         |
| opencv-python      | 4.10.0                 | Provides `cv2`.                         |
| numpy              | 1.26.4                 | Array math.                             |

### 2.1 Linux

```bash
# Optional system libs (Ubuntu/Debian):
sudo apt update && sudo apt install -y libgl1  # needed by OpenCV’s high-gui backend

# CPU-only install
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install facenet-pytorch opencv-python numpy
```

*GPU install*: replace the first `pip install torch …` line with the CUDA wheel from the PyTorch “Get Started” selector for your driver/toolkit.

### 2.2 macOS (ARM & Intel)

```bash
# Homebrew OpenCV backend (optional but speeds up video windows)
brew install opencv

pip install --upgrade pip
pip install torch torchvision torchaudio   # pulls the universal2 wheel
pip install facenet-pytorch opencv-python numpy
```

### 2.3 Windows (10/11)

```powershell
py -m pip install --upgrade pip
# CPU-only
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
py -m pip install facenet-pytorch opencv-python numpy
```

For NVIDIA GPUs, pick the CUDA wheel from the official PyTorch site and install it instead of the CPU wheel.

---

## 3 . Running the demos

| Script               | Purpose                                   | Run command                              |
|----------------------|-------------------------------------------|------------------------------------------|
| **00_camtest.py**    | Raw webcam preview                        | `python 00_camtest.py`                   |
| **01_detection.py**  | Real-time face detection + landmarks      | `python 01_detection.py`                 |
| **02_blurring.py**   | Automatic face-blurring (GDPR safe cam)   | `python 02_blurring.py`                  |
| **test.py**          | Lists available camera indices            | `python test.py`                         |

**Keys**

* Press **q** in any video window to exit.  
* If no window appears, run `python test.py` first to confirm which camera indices are available.

---

## 4 . Troubleshooting

* **`cv2.error: (-2:Unspecified error) The function is not implemented`** — install `libgl1` (Linux) or brew OpenCV (macOS) so OpenCV can create UI windows.  
* **`RuntimeError: no camera`** — another application may be locking the webcam; close it or choose another index in the code.  
* **Slow FPS on GPU** — build facenet-pytorch with `OMP_NUM_THREADS=1` and ensure you launched the CUDA-enabled wheel of torch.

---