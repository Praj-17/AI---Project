
import speech_recognition as sr
from Features.speak import speak
# from speak import speak
from serpapi import GoogleSearch
#from config import serp_api_id
serp_api_id = "50efe51a6dc4385537bad7b576ae20f16c6e20bb97eafc734be4e0ac63dd4b73"
# serp_api_id =  "92634d753e34b284b752cf279deff86eadc57fb0438b0082937be71dd5c95f17"

def auto_correct(query):
    params = {
    "q": query,
    "hl": "en",
    "gl": "us",
    "api_key": serp_api_id
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    print(results)
    search_information = results['search_information']
    try:
      return search_information['spelling_fix']
    except:
        return query



def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1.2
        r.non_speaking_duration =0.3
        r.energy_threshold = 340
        audio = r.listen(source, phrase_time_limit= 6)
        
        
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language= "en-in")
        print(f" U said: {query}")
    except:
        speak("Couldn't understand, say that again please!")
        query = listen() 
    
    return (auto_correct(query)).lower()

auto_correct("he es a gret persn")