from django.shortcuts import render
import json
from django.shortcuts import render
from celery.result import AsyncResult
from django.http import HttpResponse
from taskapp.forms import GenerateRandomUserForm
from taskapp.tasks import say_hi

def test(request):
    return HttpResponse( say_hi())