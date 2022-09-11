#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:59:19 2022

@author: daewon
"""
# 온습도 센서
# ADafruit_DHT 라이브러리 사용
# sudo apt install libgpiod2
# pip install adafruit-blinka
# pip install adafruit-circuitpython-dht ==>    파이썬 온습도 모
# sudo pip install Adafruit_DHT

#set pin number 
sensorPin = 13

# set pin numbering
GPIO.setmode(GPIO.BCM)
GPIO.setup(sensorPin, GPIO.OUT, initial = GPIO.HIGH)

/Users/daewon/.spyder-py3