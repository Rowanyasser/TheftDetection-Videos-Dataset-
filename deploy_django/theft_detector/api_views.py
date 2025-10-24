"""
REST API views for the Theft Detection system
"""
import os
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from .serializers import VideoUploadSerializer, PredictionResponseSerializer, ErrorResponseSerializer
from .utils import load_model, predict_video, get_suspicious_frames
from .views import get_model


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def api_predict(request):
    """
    REST API endpoint for video theft detection
    
    Accepts a video file via multipart/form-data and returns a JSON response
    with the prediction, confidence, and suspicious frames (if shoplifting detected).
    
    **Request:**
    - Method: POST
    - Content-Type: multipart/form-data
    - Body: video file (mp4, avi, mov, etc.)
    
    **Response:**
    ```json
    {
        "success": true,
        "prediction": 1,
        "probability": 0.89,
        "confidence": 89.0,
        "label": "Shoplifting",
        "is_theft": true,
        "suspicious_frames": [
            "/media/uploads/video_suspect_1_t5_abc123.jpg",
            "/media/uploads/video_suspect_2_t3_def456.jpg",
            "/media/uploads/video_suspect_3_t7_ghi789.jpg"
        ],
        "message": "Shoplifting detected with 89.0% confidence"
    }
    ```
    
    **Error Response:**
    ```json
    {
        "success": false,
        "error": "Error message describing the issue"
    }
    ```
    """
    # Validate request data
    serializer = VideoUploadSerializer(data=request.data)
    if not serializer.is_valid():
        error_serializer = ErrorResponseSerializer(data={
            'success': False,
            'error': str(serializer.errors)
        })
        error_serializer.is_valid()
        return Response(error_serializer.data, status=status.HTTP_400_BAD_REQUEST)
    
    video_file = serializer.validated_data['video']
    
    # Save uploaded video temporarily
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, video_file.name)
    
    try:
        # Write video file
        with open(file_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)
        
        # Get prediction
        model = get_model()
        result = predict_video(file_path, model)
        
        # Build response data
        response_data = {
            'success': True,
            'prediction': result['prediction'],
            'probability': round(result['probability'], 4),
            'confidence': round(result['confidence'] * 100, 2),
            'label': result['label'],
            'is_theft': result['prediction'] == 1,
        }
        
        # Extract suspicious frames if shoplifting detected
        suspicious_urls = []
        if result['prediction'] == 1:
            frames_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            saved_paths = get_suspicious_frames(file_path, model, k=3, save_dir=frames_dir)
            
            # Convert absolute paths to relative URLs
            for p in saved_paths:
                rel = os.path.relpath(p, settings.MEDIA_ROOT)
                url = os.path.join(settings.MEDIA_URL, rel).replace('\\', '/')
                suspicious_urls.append(url)
            
            response_data['suspicious_frames'] = suspicious_urls
            response_data['message'] = f"Shoplifting detected with {response_data['confidence']:.1f}% confidence"
        else:
            response_data['message'] = f"No shoplifting detected ({response_data['confidence']:.1f}% confidence)"
        
        # Clean up uploaded video (keep extracted frames)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Validate and return response
        response_serializer = PredictionResponseSerializer(data=response_data)
        response_serializer.is_valid(raise_exception=True)
        
        return Response(response_serializer.data, status=status.HTTP_200_OK)
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        error_serializer = ErrorResponseSerializer(data={
            'success': False,
            'error': f'Error processing video: {str(e)}'
        })
        error_serializer.is_valid()
        return Response(error_serializer.data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def api_health(request):
    """
    Health check endpoint
    
    Returns the status of the API and model availability.
    
    **Response:**
    ```json
    {
        "status": "healthy",
        "model_loaded": true,
        "message": "Theft Detection API is running"
    }
    ```
    """
    try:
        model = get_model()
        model_loaded = model is not None
    except:
        model_loaded = False
    
    return Response({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'message': 'Theft Detection API is running' if model_loaded else 'Model not loaded'
    }, status=status.HTTP_200_OK)
