from django.conf.urls import url
from django.contrib import admin
from django.urls import path, include
import data.views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    path('', data.views.index, name='index'),
    path('articles/', data.views.articles, name='articles'),
    path('articles/<int:id>/', data.views.article, name='article'),
    path('questions/', data.views.questions, name='questions'),
    path('questions/<int:id>/', data.views.question, name='question'),
    path('methods/', data.views.methods, name='methods'),
    path('methods/<int:id>/', data.views.method, name='method'),
]
