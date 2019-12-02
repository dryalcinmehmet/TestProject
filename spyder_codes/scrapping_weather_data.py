#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:00:24 2019

@author: doctor
"""

import os
import random

class WeatherData():
    
    def Scrapping():
        temp= [random.randint(25,30) for i in range(0,24)]
        humidity= [random.randint(10,20) for i in range(0,24)]
        
        return temp,humidity
        
        