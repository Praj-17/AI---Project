from Features.speak import speak
import datetime

def wishMe():
    hour=datetime.datetime.now().hour
    if hour>=0 and hour<12:
        speak("Good Morning sir, What can I do for you ?")
        print("Good Morning sir, What can I do for you ?")
    elif hour>=12 and hour<18:
        speak("Good Afternoon, What can I do for you ?")
        print("Good Afternoon, What can I do for you ?")
    else:
        speak("Good Evening, What can I do for you ?")
        print("Good Evening, What can I do for you ?")

