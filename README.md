# Shop Theft Detection — Notebooks + Django App

This repository contains two training notebooks and a Django web app to run real-time predictions on uploaded videos using a trained PyTorch model.

- TheftDetection.ipynb — Pretrained video models (r3d_18, mc3_18, r2plus1d_18) with data handling, training, and evaluation.
- TheftDetectionBaselineModel.ipynb — From-scratch baselines (CNN-LSTM and CNN+Attention) with simplified pipelines.
- deploy_django/ — Django web app for uploading a video and getting a prediction using the trained checkpoint `best_model.pth`.

best_model.pth (119 MB) is expected at the project root: `/mnt/d/Cellula_Internship/Task3/best_model.pth`.

## Repo layout

```
Task3/
  README.md                  <- this file
  requirements.txt           <- unified deps for notebooks + Django
  best_model.pth             <- trained checkpoint used by the web app
  TheftDetection.ipynb       <- pretrained models training
  TheftDetectionBaselineModel.ipynb <- baseline models training
  deploy_django/             <- Django deployment
```

## Environment setup

Create/activate a virtual environment (optional) and install dependencies from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- PyTorch installs CPU/CUDA wheels automatically based on your platform. If you need a specific CUDA build, follow the official PyTorch instructions.
- OpenCV might require system libraries on some Linux distros; if you face issues, try `pip install opencv-python-headless`.

## Datasets

Both notebooks expect a folder named `Shop Dataset` with two subfolders:

```
Shop Dataset/
  non shop lifters/
  shop lifters/
```

Path notes:
- In `TheftDetection.ipynb`, the default base path is `/mnt/d/Cellula_Internship/Task3/Shop Dataset` (WSL/Linux style).
- In `TheftDetectionBaselineModel.ipynb`, the base path is `/mnt/c/Users/Rowan/Desktop/Cellula_Internship/Task3/Shop Dataset`.
- Adjust the `base_path` variable in the first data setup cell of each notebook to match your machine.

## Trained model compatibility

The Django app expects a model trained with `r2plus1d_18` (R(2+1)D). The checkpoint key patterns (e.g., `model.stem.0`, `layer1.0.conv1.0.*`) match R(2+1)D, not plain R3D. If you train different architectures, update `deploy_django/theft_detector/utils.py` accordingly.

## Run the web app

From the project root:

```bash
cd deploy_django
export DJANGO_SETTINGS_MODULE=theft_detector.settings
python3 manage.py runserver 0.0.0.0:8000
```

Then open http://localhost:8000 and upload a video. The app will:
- Sample 8 frames at 96x96
- Run the `r2plus1d_18` model
- Return label and confidence with a modern UI
- Display suspicious frames if shoplifting is detected

If you encounter a model loading error, ensure `best_model.pth` exists at the root and that it was trained using `r2plus1d_18`.

## REST API

The deployment now includes a REST API for programmatic access to the theft detection model.

### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/api/health/
```

**Predict Video:**
```bash
curl -X POST http://localhost:8000/api/predict/ \
  -F "video=@/path/to/video.mp4"
```

**Example Response:**
```json
{
  "success": true,
  "prediction": 1,
  "probability": 0.9997,
  "confidence": 99.97,
  "label": "Shoplifting Detected",
  "is_theft": true,
  "suspicious_frames": [
    "/media/uploads/video_suspect_1_t4_abc123.jpg",
    "/media/uploads/video_suspect_2_t6_abc456.jpg",
    "/media/uploads/video_suspect_3_t0_def789.jpg"
  ],
  "message": "Shoplifting detected with 100.0% confidence"
}
```

### Python Example

```python
import requests

# Check API health
response = requests.get('http://localhost:8000/api/health/')
print(response.json())

# Predict video
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/predict/',
        files={'video': f}
    )
    result = response.json()
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2f}%")
```

For complete API documentation, see `deploy_django/API_DOCUMENTATION.md`.

## Quick test (optional)

You can run a tiny check that the model loads:

```bash
cd deploy_django
python3 test_model_load.py
```

## Troubleshooting

- Negative Dice loss in training: ensure the Dice formula parentheses are correct, as fixed in `TheftDetection.ipynb`.
- Device/CUDA: the app prints `cuda:0` if GPU is used; otherwise it runs on CPU.
- Path errors on Windows/WSL: prefer absolute paths like `/mnt/d/...`. Update `base_path` in the notebooks as needed.

