from django.shortcuts import render
from django.http import HttpResponse
import json 

def index(request):    
    return render(request,"homeApp/index.html")

def waterTemp(request):
    if(request.method=="POST"):        
        body=json.loads(request.body)
        print(body['evaluation'])      
    context={"image":"../../media/2022-08-25_01-41.jpg"}
    return render(request,"homeApp/waterTemp.html",context)

def sdo(request):
    return render(request,"homeApp/sdo.html")