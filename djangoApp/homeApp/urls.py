from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("waterTemp", views.waterTemp, name="waterTemp"),
    path("download/<str:file>", views.download_file,name='download_file'),
    path("sdo", views.sdo, name="sdo"),
    path("sdo/input", views.sdoInput, name="sdoInput"),
    path("sdo/predict", views.sdoPredict, name="sdoPredict"),
]