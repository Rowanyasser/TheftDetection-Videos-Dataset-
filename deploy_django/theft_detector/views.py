import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .utils import load_model, predict_video, get_suspicious_frames

# Load model once at startup
model = None

def get_model():
    global model
    if model is None:
        model = load_model()
    return model


def index(request):
    """Render the main upload page"""
    return render(request, 'index.html')


@csrf_exempt
def predict(request):
    """Handle video upload and return prediction"""
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        
        # Save uploaded video
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, video_file.name)
        
        with open(file_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)
        
        # Get prediction
        try:
            model = get_model()
            result = predict_video(file_path, model)

            suspicious_urls = []
            if result['prediction'] == 1:
                # Save top-K suspicious frames into MEDIA_ROOT/uploads and build URLs
                frames_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
                saved_paths = get_suspicious_frames(file_path, model, k=3, save_dir=frames_dir)
                for p in saved_paths:
                    # Convert absolute path to URL under MEDIA_URL
                    rel = os.path.relpath(p, settings.MEDIA_ROOT)
                    suspicious_urls.append(os.path.join(settings.MEDIA_URL, rel).replace('\\', '/'))

            # Optionally clean uploaded file (keep frames)
            if os.path.exists(file_path):
                os.remove(file_path)

            return render(request, 'result.html', {
                'prediction': result['label'],
                'confidence': f"{result['confidence'] * 100:.1f}%",
                'probability': result['probability'],
                'is_theft': result['prediction'] == 1,
                'suspicious_frames': suspicious_urls
            })
        except Exception as e:
            return render(request, 'result.html', {
                'error': f'Error processing video: {str(e)}'
            })
    
    return render(request, 'index.html', {'error': 'Please upload a video file'})
