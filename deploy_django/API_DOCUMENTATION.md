# REST API Documentation

## Overview

The Theft Detection REST API provides programmatic access to the shoplifting detection model. You can upload videos and receive JSON responses with predictions, confidence scores, and suspicious frame URLs.

## Base URL

```
http://localhost:8000/api/
```

## Endpoints

### 1. Health Check

Check if the API is running and the model is loaded.

**Endpoint:** `GET /api/health/`

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "message": "Theft Detection API is running"
}
```

**Example:**
```bash
curl http://localhost:8000/api/health/
```

---

### 2. Video Prediction

Upload a video for theft detection analysis.

**Endpoint:** `POST /api/predict/`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `video` (required): Video file (mp4, avi, mov, mkv, flv, wmv)
  - Maximum size: 100MB
  - Recommended: Clear footage, 5-60 seconds

**Response (Shoplifting Detected):**
```json
{
    "success": true,
    "prediction": 1,
    "probability": 0.8923,
    "confidence": 89.23,
    "label": "Shoplifting",
    "is_theft": true,
    "suspicious_frames": [
        "/media/uploads/video_suspect_1_t5_abc123.jpg",
        "/media/uploads/video_suspect_2_t3_def456.jpg",
        "/media/uploads/video_suspect_3_t7_ghi789.jpg"
    ],
    "message": "Shoplifting detected with 89.2% confidence"
}
```

**Response (No Shoplifting):**
```json
{
    "success": true,
    "prediction": 0,
    "probability": 0.1234,
    "confidence": 12.34,
    "label": "Non-Shoplifting",
    "is_theft": false,
    "message": "No shoplifting detected (87.7% confidence)"
}
```

**Error Response:**
```json
{
    "success": false,
    "error": "Error message describing the issue"
}
```

**Status Codes:**
- `200 OK`: Successful prediction
- `400 Bad Request`: Invalid input (wrong file type, file too large, etc.)
- `500 Internal Server Error`: Processing error

---

## Usage Examples

### cURL

**Health Check:**
```bash
curl http://localhost:8000/api/health/
```

**Video Prediction:**
```bash
curl -X POST \
  http://localhost:8000/api/predict/ \
  -F "video=@/path/to/your/video.mp4"
```

**Save Response to File:**
```bash
curl -X POST \
  http://localhost:8000/api/predict/ \
  -F "video=@/path/to/video.mp4" \
  -o result.json
```

---

### Python (requests)

```python
import requests

# Health check
response = requests.get('http://localhost:8000/api/health/')
print(response.json())

# Predict video
with open('shoplifting_video.mp4', 'rb') as f:
    files = {'video': f}
    response = requests.post('http://localhost:8000/api/predict/', files=files)
    
result = response.json()

if result['success']:
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    
    if result['is_theft']:
        print(f"\nSuspicious frames:")
        for frame_url in result['suspicious_frames']:
            print(f"  - http://localhost:8000{frame_url}")
else:
    print(f"Error: {result['error']}")
```

---

### JavaScript (fetch)

```javascript
// Health check
fetch('http://localhost:8000/api/health/')
  .then(response => response.json())
  .then(data => console.log(data));

// Predict video
const formData = new FormData();
formData.append('video', videoFile);

fetch('http://localhost:8000/api/predict/', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(result => {
    if (result.success) {
      console.log(`Prediction: ${result.label}`);
      console.log(`Confidence: ${result.confidence}%`);
      
      if (result.is_theft) {
        console.log('Suspicious frames:', result.suspicious_frames);
      }
    } else {
      console.error('Error:', result.error);
    }
  });
```

---

### Python (Full Example Script)

```python
#!/usr/bin/env python3
"""
Example script to test the Theft Detection REST API
"""
import requests
import json
import sys

API_BASE_URL = 'http://localhost:8000/api'

def check_health():
    """Check API health status"""
    print("Checking API health...")
    response = requests.get(f'{API_BASE_URL}/health/')
    health = response.json()
    print(f"Status: {health['status']}")
    print(f"Model loaded: {health['model_loaded']}")
    print(f"Message: {health['message']}\n")
    return health['model_loaded']

def predict_video(video_path):
    """Predict shoplifting in a video"""
    print(f"Analyzing video: {video_path}")
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            response = requests.post(f'{API_BASE_URL}/predict/', files=files)
        
        result = response.json()
        
        if result['success']:
            print(f"\n{'='*50}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*50}")
            print(f"Result: {result['label']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Probability: {result['probability']:.4f}")
            print(f"Message: {result['message']}")
            
            if result['is_theft'] and result.get('suspicious_frames'):
                print(f"\nSuspicious Frames ({len(result['suspicious_frames'])}):")
                for i, frame_url in enumerate(result['suspicious_frames'], 1):
                    full_url = f"http://localhost:8000{frame_url}"
                    print(f"  {i}. {full_url}")
            
            print(f"{'='*50}\n")
            return result
        else:
            print(f"Error: {result['error']}")
            return None
            
    except FileNotFoundError:
        print(f"Error: Video file not found: {video_path}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == '__main__':
    # Check health
    if not check_health():
        print("API is not ready. Please check the server.")
        sys.exit(1)
    
    # Predict video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        predict_video(video_path)
    else:
        print("Usage: python test_api.py <path_to_video.mp4>")
```

Save as `test_api.py` and run:
```bash
python test_api.py /path/to/video.mp4
```

---

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the request was processed successfully |
| `prediction` | integer | Binary prediction (0 = non-shoplifting, 1 = shoplifting) |
| `probability` | float | Raw model output probability (0.0 - 1.0) |
| `confidence` | float | Confidence percentage (0.0 - 100.0) |
| `label` | string | Human-readable label ("Non-Shoplifting" or "Shoplifting") |
| `is_theft` | boolean | True if shoplifting detected, False otherwise |
| `suspicious_frames` | array | URLs to suspicious frame images (only if `is_theft` is true) |
| `message` | string | Descriptive message about the prediction |
| `error` | string | Error message (only present when `success` is false) |

---

## File Constraints

- **Supported formats:** mp4, avi, mov, mkv, flv, wmv
- **Maximum file size:** 100MB
- **Recommended duration:** 5-60 seconds
- **Video quality:** Clear footage with visible subjects

---

## Notes

- The API extracts the top 3 most suspicious frames when shoplifting is detected
- Frame URLs are relative to the server's media path
- Uploaded videos are deleted after processing; only extracted frames are kept
- The same model used in the web interface powers the API
- For production use, consider adding authentication and rate limiting

---

## Troubleshooting

### Connection Refused
Ensure the Django server is running:
```bash
python3 manage.py runserver 0.0.0.0:8000
```

### Model Not Loaded
Check that `best_model.pth` exists in the parent directory and is the correct R(2+1)D checkpoint.

### File Too Large
Reduce video size or duration. The 100MB limit is configurable in `serializers.py`.

### Invalid File Type
Ensure the video has a supported extension (.mp4, .avi, .mov, etc.).

---

## Web Interface vs API

| Feature | Web Interface | REST API |
|---------|--------------|----------|
| **Access** | Browser (http://localhost:8000) | HTTP clients (curl, Python, JS) |
| **Input** | Form upload | Multipart/form-data POST |
| **Output** | HTML page with visuals | JSON response |
| **Frames** | Displayed in browser | URLs returned |
| **Use Case** | Manual testing, demos | Integration, automation |

Both interfaces use the same backend model and processing pipeline.
