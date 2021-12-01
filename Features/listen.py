
import speech_recognition as sr
from Features.speak import speak

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
        speak("Couldn't understand, say that again, please!")
        query = listen()  #recursion
    return query.lower()
