from django.db import models
from django.utils import timezone
# Create your models here.


class Topsection(models.Model):

      title = models.CharField(max_length=100)
      desc = models.TextField()
      img = models.ImageField(upload_to='pics')
      
      def __str__(self):
          return self.title

class Services(models.Model):

      title = models.CharField(max_length=100)
      desc = models.TextField()
      img = models.ImageField(upload_to='pics')
      def __str__(self):
          return self.title 


class Navhome(models.Model):
      
      title = models.CharField(max_length=100)
      desc = models.TextField(null=True)
     

      def __str__(self):
          return self.title


class News(models.Model):
      
      title = models.CharField(max_length=100)
      desc = models.TextField()
      img = models.ImageField(upload_to='pics')
      date_posted = models.DateTimeField(default=timezone.now)

      def __str__(self):
          return self.title

'''

class Navabout(models.Model):

      title = models.CharField(max_length=100)
      desc = models.TextField()
      img = models.ImageField(upload_to='pics')
      
      def __str__(self):
          return self.title          
'''

class About(models.Model):

      title = models.CharField(max_length=100)
      desc = models.TextField()
      img = models.ImageField(upload_to='pics')
      
      def __str__(self):
          return self.title



class doctor(models.Model):
      title = models.CharField(max_length=100)
      
      desc = models.TextField()
      img = models.ImageField(upload_to='pics')
     
      def __str__(self):
          return self.title   
         