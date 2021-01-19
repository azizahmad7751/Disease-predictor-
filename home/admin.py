from django.contrib import admin
from .models import Topsection,Services,News,Navhome,About,doctor
# Register your models here.

admin.site.register(Topsection),
admin.site.register(Services),
admin.site.register(News),
admin.site.register(Navhome),
admin.site.register(About),
admin.site.register(doctor)

