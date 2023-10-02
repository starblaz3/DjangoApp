from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import json 
import os
import csv
from django.conf import settings
from .MLmodel import api
from .sdo import apiSdo

def index(request):    
    return render(request,"homeApp/index.html")

def waterTemp(request):
    if(request.method=="POST"):            
        fileP=os.path.join(settings.FILES_DIR,'data.csv')
        with open(fileP, mode='wb+') as dest:                                    
            for chunk in request.FILES['csvFile'].chunks():                
                dest.write(chunk)             
        evalString=api(request.POST['model'],request.POST['evaluation'],settings.FILES_DIR)        
        return JsonResponse({'evalString': evalString}, status=200)
        # return render(request,"homeApp/waterTemp.html") 
    else:                
        return render(request,"homeApp/waterTemp.html")

def download_file(request,file):
    fileP=os.path.join(settings.FILES_DIR,file)
    if os.path.exists(fileP):
        with open(fileP,'rb') as dest:
            response=HttpResponse(dest.read(),content_type="text/csv")
            response['Content-Disposition']='attachment; filename='+str(file)
            return response
    raise "Http404"

def sdo(request):    
    return render(request,"homeApp/sdo.html")

def sdoInput(request):
    if(request.method=="POST"):   
        try:     
            fileP=os.path.join(settings.FILES_DIR,'sdo.csv')
            with open(fileP, mode='wb+') as dest:                                    
                for chunk in request.FILES['csvFile'].chunks():                
                    dest.write(chunk)             
            apiSdo(settings.FILES_DIR)
            return JsonResponse({'message': 'success'}, status=200)    
        except:
            return JsonResponse({'error': repr(e)}, status=500)
    return render(request,"homeApp/sdoInput.html")

def sdoPredict(request):
    if(request.method=="POST"):   
        try:     
            fileP=os.path.join(settings.FILES_DIR,'sdo.csv')        
            with open(fileP, mode='wb+') as dest:                                                  
                for chunk in request.FILES['csvFile'].chunks():                                    
                    dest.write(chunk)                         
            apiSdo(settings.FILES_DIR)                   
            return JsonResponse({'message': 'success'}, status=200)            
        except Exception as e:
            return JsonResponse({'error': repr(e)}, status=500)
    return render(request,"homeApp/sdoPredict.html")