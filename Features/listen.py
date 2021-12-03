
import speech_recognition as sr
from Features.speak import speak
# from speak import speak
from serpapi import GoogleSearch
from config import serp_api_id


def auto_correct(query):
    params = {
    "q": query,
    "hl": "en",
    "gl": "us",
    "api_key": serp_api_id
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    search_information = results['search_information']
    try:
      return search_information['spelling_fix']
    except:
        return query



def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.energy_threshold = 350
        audio = r.listen(source, phrase_time_limit= 4)
        
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language= "en-in")
        print(f"U said: {query}")
    except:
        speak("Couldn't understand, say that again please!")
        query = listen() 
    
    return (auto_correct(query)).lower()
