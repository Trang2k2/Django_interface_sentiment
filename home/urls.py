#from django.urls import path

#from . import views

#urlpatterns = [
#    path('', views.index, name='index'),
#]
# trong file urls.py của ứng dụng
# sentiment_analysis_app/urls.py

from django.urls import path
from .views import home, index1

urlpatterns = [
    path('', home, name='home'),
    path('index1/', index1, name='index1'),
]




