# import pyttsx3
# engine = pyttsx3.init("sapi5") #google API
# voices = engine.getProperty(('voices'))
# engine.setProperty('voices', voices[0].id)
# engine.setProperty('rate',170 )  # rate by default is 200
import pyttsx3
def speak(audio):
    engine = pyttsx3.init("sapi5") #google API
    voices = engine.getProperty(('voices'))
    engine.setProperty('voices', voices[0].id)
    engine.setProperty('rate',170 )  # rate by default is 200
    print(f"A.I : {audio}")
    engine.say(text = audio)
    engine.runAndWait()
    print(" ")

    