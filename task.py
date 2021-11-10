#2 types of function
# 1 - Non input
#eg: time, date,speedtest

# 2 - Input
# google search, wikipedia

#First we will have to create functions and then add them to the json file
import random
from pyjokes.pyjokes import get_joke
from Features import news
import datetime
from Features.listen import listen
from Features.speak import speak
import wikipedia
import pywhatkit
from Features import joke
from Features.alarm import  set_alarm
from Features.weather import  weather, weather_updates , Location
from word2number import w2n
import time
import os

import csv

def Time():
    time = datetime.datetime.now().strftime("%H: %M")
    speak(time)
def Date():
    date = datetime.date.today()
    speak(date)
def Day():
    day = datetime.datetime.now().strftime("%A")
    speak(day) 
def NEWS():
    news_res = news.get_news()
    responses = ["news","Headlines are ","Top 5 bulletins are ", 'Todays Headlines are..']
     
    speak('Source: The Times Of India')
    speak(random.choice(responses))
    for index, articles in enumerate(news_res):
        print(articles['title'])
        speak(articles['title'])
        if index == len(news_res)-2:
            break
    speak('These were the top headlines, Have a nice day Sir!!..')
    
def read_prev_response():
    lis = list(csv.reader(open('data.csv')))
    l = lis[-1]
    prev_response = str(l[-1])
    speak(prev_response)
    return prev_response
def prev_response():
    lis = list(csv.reader(open('data.csv')))
    l = lis[-1]
    prev_response = str(l[-1])
    # speak(prev_response)
    return prev_response
def final_weather():
    weather()
    speak("Do you want to listen more in detail?")
    ans = listen()
    if ans == "yes":
        weather_updates() 
def wait(amt=10):
    speak("ok, i'll wait for 10 secs")
    time.sleep(amt)
    speak("do you want me to wait more?")
    reply = listen()
    if reply == 'yes' or reply == "yeah" or "yep":
        speak("please tell, how many seconds should i wait?")
        amt = w2n.word_to_num((listen()))
        speak(f"ok ,i'll wait for {amt} seconds")
        time.sleep(amt)
    elif reply == 'no' or  reply == 'not' or reply == 'nope' or reply == 'na':
        time.sleep(0)
    else: time.sleep(0)
def location(query):
    said = query.split(" ")
    location = said[1] #obtain place name
    speak("Hold on, I will show you where {location} is")
    os.system("google-chrome https://www.google.nl/maps/place/" + location)
    speak("Here it is....")

def InputExecution(tag, query):
    if "wikipedia" in  tag:
        result = wikipedia.summary(query, sentences = 5)
        speak(result)    
    elif "google" in tag:
        query = str(query).replace("google", "").replace("search", "").replace("","").replace("what is","").replace("search about","").replace("search for","").replace("find","")
        pywhatkit.search(query)
    elif "weather" in tag :
        final_weather()  
    elif 'play' in tag:
        song = query.replace('play', '')
        speak("ok,playing" + song)
        pywhatkit.playonyt(song)  
      
    
def NoninputExecution(query):
    query = str(query)
    
    if "time" in query:
        Time()
    elif "date" in query:
        Date()
    elif "day" in query:
        Day()
    elif "news" in query:
        NEWS()
    elif "joke" in query:
        joke.startJoke()
    elif "repeat" in query:
        read_prev_response()
    elif "alarm" in query:
        set_alarm()
    elif "loaction" in query:
        Location()
    elif "wait" in query:
        wait()
    
    elif "bye" in query :
        speak
        exit(0)
        
    

# read_prev_response()

# get_joke()
# Day()
