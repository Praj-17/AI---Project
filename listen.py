
import speech_recognition as sr

from speak import speak

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.energy_threshold = 400
        audio = r.listen(source)
        
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language= "en-in")
        print(f"U said: {query}")
    except:
        speak("Couldn't understand, say that again please!")
    query = str(query)
    return query.lower()
