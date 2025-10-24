from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views, api_views

urlpatterns = [
    # Web interface routes
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    
    # REST API routes
    path('api/predict/', api_views.api_predict, name='api_predict'),
    path('api/health/', api_views.api_health, name='api_health'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
