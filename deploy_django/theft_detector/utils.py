import torch
import os
import uuid
import numpy as np
import cv2
from torchvision.models.video import r2plus1d_18  # Changed from r3d_18
import torch.nn as nn

# Path to the trained model checkpoint
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'best_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PretrainedResNet3D(nn.Module):
    """R(2+1)D model for video classification (matching the trained checkpoint)"""
    def __init__(self):
        super(PretrainedResNet3D, self).__init__()
        # Use R(2+1)D to match the checkpoint architecture
        try:
            # PyTorch 2.x uses weights parameter
            from torchvision.models.video import R2Plus1D_18_Weights
            self.model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        except (ImportError, TypeError):
            # Fallback for PyTorch 1.x
            self.model = r2plus1d_18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)


def load_model():
    """Load the trained model from checkpoint"""
    model = PretrainedResNet3D()
    if os.path.exists(MODEL_PATH):
        # Load with strict=False to allow minor architecture differences
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
    model.to(DEVICE)
    model.eval()
    return model


def extract_frames(video_path, fixed_frames=8, frame_size=(96, 96)):
    """Extract fixed number of frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.zeros((fixed_frames, frame_size[0], frame_size[1], 3), dtype=np.float32)
    
    step = max(1, total_frames // fixed_frames)
    for i in range(fixed_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.float32))
            continue
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        frames.append(frame.astype(np.float32))
    
    cap.release()
    while len(frames) < fixed_frames:
        frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.float32))
    
    return np.array(frames)


def preprocess_video(video_path, fixed_frames=8, frame_size=(96, 96)):
    """Preprocess video for model input"""
    frames = extract_frames(video_path, fixed_frames=fixed_frames, frame_size=frame_size)
    # Convert to tensor shape (1, C, T, H, W)
    frames = np.transpose(frames, (3, 0, 1, 2))
    tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
    return tensor


def predict_video(video_path, model):
    """Run prediction on a video file"""
    input_tensor = preprocess_video(video_path)
    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).cpu().item()
        prediction = 1 if prob > 0.5 else 0
    
    return {
        'prediction': prediction,
        'probability': prob,
        'label': 'Shoplifting Detected' if prediction == 1 else 'No Shoplifting Detected',
        'confidence': prob if prediction == 1 else (1 - prob)
    }


def _sample_indices(video_path, fixed_frames=8):
    """Compute frame indices used by extract_frames for alignment with original video frames."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        return list(range(fixed_frames))
    step = max(1, total_frames // fixed_frames)
    return [i * step for i in range(fixed_frames)]


def compute_frame_importance(model, input_tensor):
    """Occlusion-based importance for each temporal frame.

    Returns a list of length T with the drop in probability when each frame is occluded.
    """
    model.eval()
    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad():
        base_logit = model(input_tensor)
        base_prob = torch.sigmoid(base_logit).item()
    T = input_tensor.shape[2]
    scores = []
    # Iterate frames and occlude one at a time
    for t in range(T):
        occluded = input_tensor.clone()
        occluded[:, :, t, :, :] = 0.0
        with torch.no_grad():
            prob = torch.sigmoid(model(occluded)).item()
        scores.append(base_prob - prob)
    return scores, base_prob


def get_suspicious_frames(video_path, model, k=3, fixed_frames=8, save_dir=None):
    """Identify and optionally save the k most suspicious frames.

    - Uses occlusion to score frame importance.
    - Reads original frames at the same sampled positions used for inference.
    - If save_dir is provided, saves JPEGs and returns a list of file paths; otherwise returns numpy arrays.
    """
    # Prepare input for scoring
    input_tensor = preprocess_video(video_path, fixed_frames=fixed_frames)
    scores, base_prob = compute_frame_importance(model, input_tensor)

    # Pick top-k indices by score
    indices = np.argsort(scores)[::-1][:k].tolist()

    # Read original frames at those indices
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = _sample_indices(video_path, fixed_frames=fixed_frames)
    raw_frames = []
    for i, idx in enumerate(frame_idxs):
        if idx >= total_frames:
            idx = total_frames - 1
        if idx < 0:
            idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
        raw_frames.append(frame)  # BGR
    cap.release()

    # Collect selected frames
    selected = [(i, raw_frames[i]) for i in indices if i < len(raw_frames)]

    # Save if requested
    outputs = []
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        uid = uuid.uuid4().hex[:8]
        for rank, (i, frame_bgr) in enumerate(selected, start=1):
            # Ensure reasonable size for display
            try:
                h, w = frame_bgr.shape[:2]
                scale = 320 / max(h, w) if max(h, w) > 320 else 1.0
                if scale != 1.0:
                    frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
            except Exception:
                pass
            out_name = f"{base}_suspect_{rank}_t{i}_{uid}.jpg"
            out_path = os.path.join(save_dir, out_name)
            cv2.imwrite(out_path, frame_bgr)
            outputs.append(out_path)
        return outputs
    else:
        return [f for _, f in selected]
