from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('prediksi/', views.prediksi, name='prediksi'),
    path('dataset/', views.dataset_view, name='dataset'),
    path('dataset/unduh/', views.unduh_dataset, name='unduh_dataset'),
]