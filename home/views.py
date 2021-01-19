from django.shortcuts import render 
from .models import Topsection, Navhome, News, About, doctor,Services
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.core.mail import send_mail
from django.conf import settings
# Create your views here.


def home(request):
    tops = Topsection.objects.all()
    nav= Navhome.objects.all()
    navdoct= doctor.objects.all()
    serv=Services.objects.all()
    new= News.objects.all()
    print(tops)
    
    #nav=Navtop.objects.all()
    return render(request,
      "home.html",
      {
      'tops': tops,
      'nav': nav, 
      'navdoct':navdoct,
      'serv':serv, 
      'new': new  
      }
    )

'''

def navhome(request):
    
    nav= Navhome.objects.all()
 #  return render(request, "home.html", {"service": service})
    print(nav)
    return render(request, 
      "home.html", 
      {
      'nav': nav 
      }
                  
                  )    
'''
'''

def services(request):

    service = Services.objects.all()
    print(service)
#return render(request, "home.html", {"service": service})
    return render(request, "home.html",
                  {'service': service}
                  )
'''
def test(request):
    
    
    return render(request, "test.html",

                 
                  )

    

def news(request):

    news= News.objects.all()

    paginator =  Paginator(news, per_page=3)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
 #  return render(request, "home.html", {"service": service})
    return render(request, 
      "news.html",
      {
      'news': page_obj.object_list,
      'paginator': paginator,
      'page_number': int(page_number)

      }
    )
'''

def Navabout(request):
    
    about= Navabout.objects.all()
 #  return render(request, "home.html", {"service": service})
    print(about)
    return render(request, "about.html", 
                  {'about': about }
                  )

'''
def about(request):
    
    abouts= About.objects.all()
    navdoc= doctor.objects.all()
 #  return render(request, "home.html", {"service": service})
    print(about)
    return render(request, "about.html", 
                  {'abouts': abouts,
                   'navdoc':navdoc


                                        }
                  )
 #  return render(request, "home.html", {"service": service})
  


  
def contact(request):

    if request.method == 'POST':

        fname= request.POST['fname']
        email= request.POST['email']
        message= request.POST['message']
        send_mail(
          fname,
          message,
          settings.EMAIL_HOST_USER,
          ['caps7751@gmail.com'],
          
          )

        mess='Submitted Successfully'  
    #  return render(request, "home.html", {"service": service})
        return render(request, "contact.html",{'mess':mess})    
    else:
        return render(request, 'contact.html',{})



 

#def registerdlabs(request):

 # return render(request, "home.html", {"service": service})
   # return render(request, "reglabs.html")



#APPLICATION DISEASE PREDICTOR


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
#from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import Adam
#from keras.layers import Dense
#from keras.optimizers import Adam
#from keras.layers import Dropout

@login_required(login_url='accounts/login')
def heart(request):
  """ 
  18:39:18 09 Oct, 2019 by Aziz Ahmad
  Reading the training data set. 
  """
  df = pd.read_csv('static/Datasets/heart.csv')
  data = df.values
  X = data[:, :-1]
  Y = data[:, -1:]

  """ 
    18:39:18 09 Oct, 2019 by Aziz Ahmad
    Reading data from the user. 
    """

  value = ''

  if request.method == 'POST':

    age = float(request.POST['age'])
    sex = float(request.POST['sex'])
    cp = float(request.POST['cp'])
    trestbps = float(request.POST['trestbps'])
    chol = float(request.POST['chol'])
    fbs = float(request.POST['fbs'])
    restecg = float(request.POST['restecg'])
    thalach = float(request.POST['thalach'])
    exang = float(request.POST['exang'])
    oldpeak = float(request.POST['oldpeak'])
    slope = float(request.POST['slope'])
    ca = float(request.POST['ca'])
    thal = float(request.POST['thal'])

    user_data = np.array(
        (age,
         sex,
         cp,
         trestbps,
         chol,
         fbs,
         restecg,
         thalach,
         exang,
         oldpeak,
         slope,
         ca,
         thal)
    ).reshape(1, 13)
    # define the keras 
    # Define a "shallow" logistic regression model
   
    model = Sequential()
    model.add(Dense(12,input_dim=13, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics = ['accuracy'])
 
    # fit the keras model on the dataset
    model.fit(X, Y, epochs=100, batch_size=50)
    # make class predictions with the model
    predictions = model.predict_classes(user_data)
    
    
    
 
  
    if int(predictions[0]) == 1:
      value = 'have'
    elif int(predictions[0]) == 0:
      value = "don\'t have"

    return render(request,
                'heart.html',
                {    

                    'context': value,
                    'age': age,
                    'sex':sex,
                    'cp':cp,
                    'trestbps':trestbps,
                    'chol':chol,
                    'fbs':fbs,
                    'restecg':restecg,
                    'thalach':thalach,
                    'exang':exang,
                    'oldpeak':oldpeak,
                    'slope':slope,
                    'ca':ca,
                    'thal':thal

                })
  else:
    return render(request,'heart.html',{})
    

@login_required(login_url='accounts/login')
def diabetes(request):
  """ 
  20:13:20 09 Oct, 2019 by Aziz Ahmad
  Reading the training data set. 
  """
  dfx = pd.read_csv('static/Datasets/Diabetes_XTrain.csv')
  dfy = pd.read_csv('static/Datasets/Diabetes_YTrain.csv')
  X = dfx.values
  Y = dfy.values
 # Y = Y.reshape((-1,))

  """ 
    20:18:20 09 Oct, 2019 by Aziz Ahmad
    Reading data from user. 
    """
  value = ''
  if request.method == 'POST':

    pregnancies = float(request.POST['pregnancies'])
    glucose = float(request.POST['glucose'])
    bloodpressure = float(request.POST['bloodpressure'])
    skinthickness = float(request.POST['skinthickness'])
    bmi = float(request.POST['bmi'])
    insulin = float(request.POST['insulin'])
    pedigree = float(request.POST['pedigree'])
    age = float(request.POST['age'])

    user_data = np.array(
        (pregnancies,
         glucose,
         bloodpressure,
         skinthickness,
         bmi,
         insulin,
         pedigree,
         age)
    ).reshape(1, 8)

    model = Sequential()
    model.add(Dense(12,input_dim=8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics = ['accuracy'])
 
    # fit the keras model on the dataset
    model.fit(X, Y, epochs=100, batch_size=50)
    predictions = model.predict_classes(user_data)

    if int(predictions[0]) == 1:
      value = 'have'
    elif int(predictions[0]) == 0:
      value = "don\'t have"

    return render(request,
                'diabetes.html',
                {

                    'context': value,
                    'pregnancies':pregnancies,
                    'glucose':glucose,
                    'bloodpressure':bloodpressure,
                    'skinthickness':skinthickness,
                    'bmi':bmi,
                    'insulin':insulin,
                    'pedigree':pedigree,
                    'age':age

                })
  else:
    return render(request,'diabetes.html',{})
    


@login_required(login_url='accounts/login')
def breast(request):
  """ 
  20:56:20 09 Oct, 2019 by Aziz Ahmad
  Reading training data set. 
  """
  df = pd.read_csv('static/Datasets/breastcancer.csv')

  
  X = df.iloc[:,2:7].values  
  Y = df.iloc[:,1].values
  

  """ 
    20:57:20 09 Oct, 2019 by Aziz Ahmad
    Reading data from user. 
    """
  value = ''
  if request.method == 'POST':

    radius = float(request.POST['radius'])
    texture = float(request.POST['texture'])
    perimeter = float(request.POST['perimeter'])
    area = float(request.POST['area'])
    smoothness = float(request.POST['smoothness'])
    
    user_data = np.array(
        (radius,
         texture,
         perimeter,
         area,
         smoothness
         )

        ).reshape(1, 5)

    
    model = Sequential()
    model.add(Dense(12,input_dim=5, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics = ['accuracy'])
 
    # fit the keras model on the dataset
    model.fit(X, Y, epochs=100, batch_size=50)
    predictions = model.predict_classes(user_data)


    if int(predictions[0]) == 1:
      value = 'have'
    elif int(predictions[0]) == 0:
      value = "don\'t have"

    return render(request, 'breast.html',

                {
                    'context': value,
                    'radius':radius,
                    'texture':texture,
                    'perimeter':perimeter,
                    'area':area,
                    'smoothness':smoothness


                })
  else:
    return render(request,'breast.html',{})    


@login_required(login_url='accounts/login')
def index(request):
  

  tops = Topsection.objects.all()

  return render(request, "index.html",{'tops': tops })


def handler404(request):
    return render(request, '404.html', status=404)

'''
@login_required(login_url='accounts/login')
def Navdoc(request):
    
    doct= Navdoc.objects.all()
 #  return render(request, "home.html", {"service": service})
    print(about)
    return render(request, "doctors.html", 
                  {'doct': doct })

'''
@login_required(login_url='accounts/login')

def doctors(request): 

    doct= doctor.objects.all()
 #  return render(request, "home.html", {"service": service}
   
    paginator =  Paginator(doct, per_page=3)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
 #  return render(request, "home.html", {"service": service})
    return render(request, 
      "doctors.html",
      {
      'doct': page_obj.object_list,
      'paginator': paginator,
      'page_number': int(page_number)

      }  
    )  


@login_required(login_url='accounts/login')
def search(request):


          
    qu=  request.GET.get('search')      
        
    status = doctor.objects.filter(title__contains=qu)

    paginator =  Paginator(status, per_page=3)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
 #  return render(request, "home.html", {"service": service})
    return render(request, 
      "doctors.html",
      {
      'status': page_obj.object_list,
      'paginator': paginator,
      'page_number': int(page_number)

      } 
    )  
  



"""
    rf = RandomForestClassifier(
        n_estimators=16, criterion='entropy', max_depth=5)
    rf.fit(np.nan_to_num(X), Y)

    user_data = np.array((radius, texture, perimeter, area, smoothness)).reshape(1, 5)

    predictions = rf.predict(user_data)
    print(predictions)
 """ 
