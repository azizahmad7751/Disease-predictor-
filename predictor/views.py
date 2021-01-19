from django.shortcuts import render
# Create your views here.
from django.shortcuts import render

'''import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout

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
                    'context': value
                })


def diabetes(request):
  """ 
  20:13:20 09 Oct, 2019 by Aziz Ahmad
  Reading the training data set. 
  """
  dfx = pd.read_csv('static/Datasets/Diabetes_XTrain.csv')
  dfy = pd.read_csv('static/Datasets/Diabetes_YTrain.csv')
  X = dfx.values
  Y = dfy.values
  Y = Y.reshape((-1,))

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

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, Y)

    predictions = knn.predict(user_data)

    if int(predictions[0]) == 1:
      value = 'have'
    elif int(predictions[0]) == 0:
      value = "don\'t have"

  return render(request,
                'diabetes.html',
                {
                    'context': value
                })


def breast(request):
  """ 
  20:56:20 09 Oct, 2019 by Aziz Ahmad
  Reading training data set. 
  """
  df = pd.read_csv('static/Datasets/Breast_train.csv')
  data = df.values
  X = data[:, :-1]
  Y = data[:, -1]
  print(X.shape, Y.shape)

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

    """ 
        21:02:21 09 Oct, 2019 by Aziz Ahmad
        Creating our training model. 
        """
    rf = RandomForestClassifier(
        n_estimators=16, criterion='entropy', max_depth=5)
    rf.fit(np.nan_to_num(X), Y)

    user_data = np.array(
        (radius, texture, perimeter, area, smoothness)).reshape(1, 5)

    predictions = rf.predict(user_data)
    print(predictions)

    if int(predictions[0]) == 1:
      value = 'have'
    elif int(predictions[0]) == 0:
      value = "don\'t have"

  return render(request, 'breast.html',
                {
                    'context': value
                })


def index(request):

  return render(request, 'index.html')


""" 
20:07:20 10 Oct, 2019 by Aziz Ahmad
Handling 404 error pages. 
"""
def contact(request):

 #  return render(request, "home.html", {"service": service})
    return render(request, "contact.html")    

def about(request):

 #  return render(request, "home.html", {"service": service})
    return render(request, "about.html")      


def handler404(request):
  return render(request, '404.html', status=404)
'''