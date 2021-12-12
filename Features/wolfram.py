import wolframalpha
import urllib.request     #urllib. request for opening and reading URLs
import ssl
<<<<<<< HEAD
# from Features.speak import speak 
from speak import speak
=======
from Features.speak import speak 

>>>>>>> c46e833024edf52ed67b8a38dcdfc74b473e559d
#from config import wolfram_id
wolfram_id="2UR3A3-RA6XVYLJ6E"
def wolfram_ssl(query):
    client = wolframalpha.Client(wolfram_id)
    res = client.query(query)
    #speak(next(response.results).text)
    # speak(str(response))
<<<<<<< HEAD

    # write below code in task.py 
'''try:
    wolfram_ssl()
    response=app.query(query)    #Allows for arbitrary parameters to be passed the query
    # print(next(response.results).text)
    speak(next(response.results).text)
except:
    # print("Say that again,Please...")
    speak("Say that again,Please...")'''

                
=======
    answer = next(res.results).text
    speak(answer)
    print(answer)
# get_joke()
>>>>>>> c46e833024edf52ed67b8a38dcdfc74b473e559d
