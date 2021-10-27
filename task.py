#2 types of function
# 1 - Non input
#eg: time, date,speedtest

# 2 - Input
# google search, wikipedia

#First we will have to create functions and then add them to the json file
import datetime
from speak import speak
import wikipedia
import pywhatkit
def Time():
    time = datetime.datetime.now().strftime("%H: %M")
    speak(time)
def Date():
    date = datetime.date.today()
    speak(date)
def Day():
    day = datetime.datetime.now().strftime("%A")
    speak(day)
    
def NoninputExecution(query):
    query = str(query)
    
    if "time" in query:
        Time()
    elif "date" in query:
        Date()
    elif "day" in query:
        Day()
def InputExecution(tag, query):
    if "wikipedia" in  tag:
        name = str(query).replace("who is","").replace("about","").replace("tell me about","").replace("search","").replace("find","").replace("inform me about","").replace("search for","").replace("gather information on","").replace("gather info on","").replace("describe","").replace("tell about","")
        result = wikipedia.summary(name, sentences = 5)
        speak(result)
    elif "google" in tag:
        query = str(query).replace("google", "").replace("search", "").replace("","").replace("what is","").replace("search about","").replace("search for","").replace("find","")
        pywhatkit.search(query)
        
        
# Day()
