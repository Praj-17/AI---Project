import wolframalpha
from Features.speak import speak 

#from config import wolfram_id
wolfram_id="2UR3A3-RA6XVYLJ6E"
def wolfram_ssl(query):
    client = wolframalpha.Client(wolfram_id)
    res = client.query(query)
    #speak(next(response.results).text)
    # speak(str(response))
    answer = next(res.results).text
    speak(answer)


