import wolframalpha
import urllib.request     #urllib. request for opening and reading URLs
import ssl
import config
from Features.speak import speak 

try:
    app=wolframalpha.Client(config.wolfram_id)
except Exception as p:
    print("Exception: " + str(p))

def wolfram_ssl():
    ssl._create_default_https_context = ssl._create_unverified_context
    response = urllib.request.urlopen('https://www.wolframalpha.com/')
    speak(response)

    # write below code in task.py 
'''try:
    wolfram_ssl()
    response=app.query(query)    #Allows for arbitrary parameters to be passed the query
    # print(next(response.results).text)
    speak(next(response.results).text)
except:
    # print("Say that again,Please...")
    speak("Say that again,Please...")'''