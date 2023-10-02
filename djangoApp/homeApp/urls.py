from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("waterTemp", views.waterTemp, name="waterTemp"),
    path("sdo", views.sdo, name="sdo"),
]