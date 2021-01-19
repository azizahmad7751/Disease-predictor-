from django.contrib.auth.models import auth, User
from django.contrib import messages
from django.shortcuts import render, redirect
# Create your views here.
def register(request):

    if request.method == 'POST':

        username = request.POST['name']
        password1 = request.POST['password']
        password2 = request.POST['password1']
        email = request.POST['email']

        if password1 == password2:
            if username=='':
                messages.info(request, 'Username is required')
                return redirect('register') 
            elif User.objects.filter(username=username).exists(): 
                messages.info(request, 'User Taken')
                return redirect('register')
               
            elif len(password1)<5:
                messages.info(request, 'Password is too Short!!!')
                return redirect('register')
            elif email=='':
                messages.info(request, 'Email is required')
                return redirect('register')    

            elif User.objects.filter(email=email).exists() :
                messages.info(request, 'Email Taken')
                return redirect('register')
            else:
                user = User.objects.create_user(
                    password=password1, username=username,  email=email)
                user.save()
                print("User Created!!!")
                return redirect('login')
        else:
            messages.info(request, 'password not matching')
            return redirect('register')
        return redirect('/')

    else:

        return render(request, "register.html")



def login(request):

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        p = User.objects.filter(email=username)
        print(p)

        user = auth.authenticate(username=username , password=password)
        
        if user is not None:
            auth.login(request, user)
            if  request.GET.get('next',None):
                return HttpResponseRedirect(request.GET['next'])
            return redirect("index")       

        else:
            messages.info(request, 'invalid credentials!!!')
            return redirect('login')
    return render(request, 'login.html')


def logout(request):
    if request.method == 'POST':

        auth.logout(request)
        return redirect('home')     

def password_reset(request):
    
    
 #  return render(request, "home.html", {"service": service})
    
    return render(request, "password_reset.html",)


                
               