import random
import json
from sklearn import preprocessing
from Neural_Network.text_preprocessing import ignore_words, preprocessing
from Features.csv_writer import append_data, prev_response, prev_time
from Features.wishme import wishMe
import numpy as np
from Features.listen import listen , listen_std
from Features.speak import speak
from task import InputExecution, NoninputExecution
from Features.wolfram import wolfram_ssl
import datetime
from datetime import datetime 
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
try:
    import pywhatkit
except Exception as e:
    print("Exception: ", e)
    speak("I'm having trouble connecting to the internet")
from Train_2 import data

with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)
# loading model
chatbot_model = load_model("TrainData")
# fitting TfIdfVectorizer with training data to preprocess inputs
data["Patterns"] = data["Patterns"].str.lower()
vectorizer = TfidfVectorizer(ngram_range=(1, 2),stop_words = ignore_words)
vectorizer.fit(data["Patterns"])

# fitting LabelEncoder with target variable(tags) for inverse transformation of predictions
le = LabelEncoder()
le.fit(data["tags"])

# transforming input and predicting intent
def predict_tag(inp_str):
    inp_data_tfidf = vectorizer.transform([preprocessing(inp_str)]).toarray()
    print(preprocessing(inp_str))
    predicted_proba =chatbot_model.predict(inp_data_tfidf)
    probability = np.max(predicted_proba)
    print (probability)
    encoded_label = [np.argmax(predicted_proba)]
    predicted_tag = le.inverse_transform(encoded_label)[0]
    return predicted_tag, probability

print(predict_tag("what an amazing thing you are"))
# def inactive():
#         time_now = datetime.now().strftime("%H:%M:%S")  
#         time_prev = prev_time()
#         FMT = '%H:%M:%S'
#         print(time_now)
#         print(time_prev)
#         difference =(datetime.strptime(time_now, FMT) - datetime.strptime(time_prev, FMT))
#         print("__________________Time Difference_______________")
#         print(difference)
#         total_seconds_difference = difference.total_seconds()
#         print("______________Time Differnce in seconds______________")
#         print(total_seconds_difference)
        
#         if total_seconds_difference >=60:
#             print("___________The status turned False__________")
#             return False
#         else:
#             print("___________The status turned True__________")
#             return True
# def Execute_command(probability):
#     if probability>=0.7:
#         print("____________Entered 0.7 zone____________")
#         for intent in intents ['intents']:
#             if tag == "Bye" and intent["tag"] == "Bye":
#                         reply = random.choice(intent["responses"])  
#                         speak(reply)
#                         append_data('data.csv',result, reply)
#                         exit(0)
#             elif tag == "repeat" and intent["tag"] == "repeat":
#                         prev_response('response')
#                         append_data('data.csv',result,prev_response())
#             elif tag == intent["tag"]:
#                     reply = random.choice(intent["responses"])
#                     append_data('data.csv',result, reply)
#                     print("__________________Final_reply_______________")
#                     print(reply)
#                     if "time" in reply:
#                         NoninputExecution(reply)
#                     elif "date" in reply:
#                         NoninputExecution(reply)
#                     elif "day" in reply:
#                         NoninputExecution(reply)
#                     elif "news" in reply:
#                         NoninputExecution(reply)
#                     elif "joke" in reply:
#                         NoninputExecution(reply)
#                     elif "alarm" in reply:
#                         NoninputExecution(reply)
#                     elif "wait" in reply:
#                         status = False
#                     elif "wikipedia" in reply:
#                         InputExecution(reply, result)
#                     elif "google" in reply:
#                         InputExecution(reply, result)
#                     elif "weather" in reply:
#                         InputExecution(reply, result)
#                     elif "location" in reply:
#                         InputExecution(reply, result)
#                     elif "playmusic" in reply:
#                         InputExecution(reply, result)
#                     else: 
#                         speak(reply)
#             #quick_reply()
#     elif probability>=0.3 and probability<0.7:
#             print("____________Entered 0.2 zone____________")
#             for intent in intents ['intents']: 
#                 if tag == intent["tag"]:
#                     reply = random.choice(intent["responses"])
#                     append_data('data.csv',result, reply)
#                     print("__________________Final_reply_______________")
#                     print(reply)
#                     if "wikipedia" in reply:
#                         InputExecution(reply, result)
#                     elif "google" in reply:
#                         InputExecution(reply, result)
#                     elif "weather" in reply:
#                         InputExecution(reply, result)
#                     elif "location" in reply:
#                         InputExecution(reply, result)
#                     elif "playmusic" in reply:
#                         InputExecution(reply, result)
#                     else:
#                             answer = wolfram_ssl(result)
#                             append_data('data.csv',result, "wlofram in 0.2 zone")
#                             if answer == "(data not available)" or answer == "StopIteration":
#                                    print("__________Data not available______")
#                                    try:
#                                        pywhatkit.search(sentence)
#                                        append_data('data.csv',sentence, "finally_googled")
#                                    except Exception as e:
#                                            print("Exception: " ,e)
#                                            speak("Please connect to the internet.")
#     else:
#             answer = wolfram_ssl(result)
#             append_data('data.csv',result, "wlofram in 0.2 zone")
#             if answer == "(data not available)" or answer == "StopIteration":
#                     print("__________Data not available______")
#                     try:
#                         pywhatkit.search(sentence)
#                         append_data('data.csv',sentence, "finally_googled")
#                     except Exception as e:
#                             print("Exception: " ,e)
#                             speak("Please connect to the internet.")
    
# if __name__ == "__main__":
#     status =False
#     wishMe()        
#     while True:
#         print("__________The status is inactive___________")
#         print("call my name (ALEXA) to resume")
#         query = listen_std()
#         if "alexa" in query:
#             status = True
#             append_data('data.csv',"Entered", "Entered")
#             while status == True:   
#                     print("__________The status is active___________")
#                     print("Ask me anything now")
#                     sentence =listen() 
#                     result = str(sentence)
#                     if result == " ":
#                         status = False
#                         break
#                     tag, probability = predict_tag(preprocessing(sentence))
#                     print 
#                     print("____________Predicted tag_________")
#                     print(tag)
#                     print("______prob_________")
#                     print(probability)   
#                     Execute_command(probability)                                   
#                     status = inactive()    