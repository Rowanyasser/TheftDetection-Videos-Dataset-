"""
Serializers for the Theft Detection REST API
"""
from rest_framework import serializers


class VideoUploadSerializer(serializers.Serializer):
    """Serializer for video upload requests"""
    video = serializers.FileField(
        required=True,
        help_text="Video file to analyze (mp4, avi, mov, etc.)"
    )
    
    def validate_video(self, value):
        """Validate that the uploaded file is a video"""
        # Check file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        file_ext = value.name.lower()[-4:]
        if not any(file_ext.endswith(ext) for ext in valid_extensions):
            raise serializers.ValidationError(
                f"Invalid file type. Supported formats: {', '.join(valid_extensions)}"
            )
        
        # Check file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if value.size > max_size:
            raise serializers.ValidationError(
                f"File too large. Maximum size is {max_size / (1024*1024):.0f}MB"
            )
        
        return value


class PredictionResponseSerializer(serializers.Serializer):
    """Serializer for prediction response"""
    success = serializers.BooleanField(
        help_text="Whether the prediction was successful"
    )
    prediction = serializers.IntegerField(
        help_text="Binary prediction: 0 (non-shoplifting) or 1 (shoplifting)"
    )
    probability = serializers.FloatField(
        help_text="Raw probability score from the model (0.0 to 1.0)"
    )
    confidence = serializers.FloatField(
        help_text="Confidence percentage (0.0 to 100.0)"
    )
    label = serializers.CharField(
        help_text="Human-readable label: 'Non-Shoplifting' or 'Shoplifting'"
    )
    is_theft = serializers.BooleanField(
        help_text="Boolean indicating if shoplifting was detected"
    )
    suspicious_frames = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="URLs of the most suspicious frames (only for shoplifting cases)"
    )
    message = serializers.CharField(
        required=False,
        help_text="Additional information or status message"
    )


class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses"""
    success = serializers.BooleanField(default=False)
    error = serializers.CharField(
        help_text="Error message describing what went wrong"
    )
