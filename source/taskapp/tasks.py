from __future__ import absolute_import, unicode_literals
import string
from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from celery import shared_task, current_task

@shared_task
def say_hi():
    print("ok")
    return("Hello Celery!")