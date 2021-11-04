
import speech_recognition as sr
from Features.speak import speak

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
    return query.lower()
