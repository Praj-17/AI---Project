#2 types of function
# 1 - Non input
#eg: time, date,speedtest

# 2 - Input
# google search, wikipedia

#First we will have to create functions and then add them to the json file
from os import read
from pyjokes.pyjokes import get_joke
from Features import news
import datetime
from Features.speak import speak
import wikipedia
import pywhatkit
from Features import joke
from Features import weather 
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
    speak('Source: The Times Of India')
    speak('Todays Headlines are..')
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
    # return prev_response
    
    
    
def InputExecution(tag, query):
    if "wikipedia" in  tag:
        result = wikipedia.summary(query, sentences = 5)
        speak(result)    
    elif "google" in tag:
        query = str(query).replace("google", "").replace("search", "").replace("","").replace("what is","").replace("search about","").replace("search for","").replace("find","")
        pywhatkit.search(query)
    
        
    
    
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
    elif "bye" in query :
        speak
        exit(0)
        
    

# read_prev_response()

# get_joke()
# Day()
