from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('prediksi/', views.prediksi, name='prediksi'),
    path('analisis/', views.analisis, name='analisis'),
    path('analisis/penjelasan/', views.analisis_penjelasan, name='analisis_penjelasan'),
    path('dataset/', views.dataset_view, name='dataset'),
    path('dataset/unduh/', views.unduh_dataset, name='unduh_dataset'),
    path('tentang/', views.tentang, name='tentang'),
]