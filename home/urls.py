from django.urls import path
from .import views

urlpatterns = [

    path('', views.home, name='home'),
    #path('services', views.services, name="services"),
    #path('Navabout', views.Navabout, name="Navabout"),
    path('about', views.about, name="about"),
    #path('navhome', views.navhome, name="navhome"),
    path('test', views.test, name="test"),
    path('news', views.news, name="news"),
    path('contact', views.contact, name="contact"),
    path('index', views.index, name='index'),
    path('heart', views.heart, name="heart"),
    path('diabetes', views.diabetes, name="diabetes"),
    path('breast', views.breast, name="breast"),
    
    path('doctors', views.doctors, name="doctors"),
    path('search', views.search, name="search")

    #path('registerdlabs', views.registerdlabs, name="registerdlabs")
]

#Navnews,Navabout,Navdoc,doctor
