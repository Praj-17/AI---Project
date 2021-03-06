import random
import json

from sklearn import preprocessing
from Neural_Network.brain import NeuralNet
from Neural_Network.text_preprocessing import stem
from Neural_Network.text_preprocessing import bag_of_words, tokenize,ignore_words, preprocessing
import torch
from Features.csv_writer import append_data, prev_response, prev_time
from Features.wishme import wishMe
from Features.reply import quick_reply
import os
import numpy as np
from Train import trained_words
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
except ImportError:
    speak("Couldn't connect to the internet")
from Train_2 import data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)

FILE = 'TrainData.pth'
if os.path.exists(FILE):
    model  = torch.load(FILE)
    model_state = model["model_state"]
    all_words = model["all_words"]
    tags = model["tags"]
    model = NeuralNet(model["input_size"],model["hidden_size"], model["output_size"]).to(device)
    model.load_state_dict( model_state)
    model.eval

else: 
    try:
        exec(open('Train.py').read())
        model  = torch.load(FILE)
    except Exception as e:
        print(e)
        print("issue with model loading")
    

#---------------------------------------------------------------------------------------------------

def inactive():
        time_now = datetime.now().strftime("%H:%M:%S")  
        time_prev = prev_time()
        FMT = '%H:%M:%S'
        print(time_now)
        print(time_prev)
        difference =(datetime.strptime(time_now, FMT) - datetime.strptime(time_prev, FMT))
        print("__________________Time Difference_______________")
        print(difference)
        total_seconds_difference = difference.total_seconds()
        print("______________Time Differnce in seconds______________")
        print(total_seconds_difference)
        
        if total_seconds_difference >=60:
            print("___________The status turned False__________")
            return False
        else:
            print("___________The status turned True__________")
            return True

           
                
if __name__ == "__main__":
    status =False
    wishMe()        

    while True:
        print("__________The status is inactive___________")
        print("call my name (ALEXA) to resume")
        query = listen_std()
        

        if "alexa" in query:
            status = True
            append_data('data.csv',"Entered", "Entered")
            while status == True:   
                    print("__________The status is active___________")
                    print("Ask me anything now")
                    sentence =listen() 
                    result = str(sentence)
                    if result == " ":
                        status = False
                        break
                   
                    
                
                    # write the differnce time code here
                    
                    #lets say sentence = "What is photosynthesis"
                    
                    sentence = tokenize(sentence)
                    sentence = [stem(w) for w in sentence if w not in ignore_words]
                    sentence = ''.join(sentence)
                    print(sentence)
                    bag_2 = np.zeros(len(trained_words), dtype = np.float32)
                    for index, w in enumerate (trained_words):
                        if w in sentence:
                            bag_2[index] = 1
                    
                    print(bag_2)
                    true_count= np.sum(bag_2)
                    print("___________True_count__________")
                    print(true_count)
                    probability = true_count/len(result)
                    print("____________probablitly_______")
                    print(probability)
                    x = bag_of_words(sentence, all_words)
                    x = x.reshape(1,x.shape[0])
                    x = torch.from_numpy(x).to(device)
                    output = model(x)
                    _, predicted = torch.max(output, dim = 1 )
                    # print(predicted)
                
                    
                    tag = tags[predicted.item()]

                    probs = torch.softmax(output, dim = 1)
                    prob = probs[0][predicted.item()]
                    
                    # if prob.item()> 0.75:
                    #     for intent in intents ['intents']:
                    print 
                    print("____________Predicted tag_________")
                    print(tag)
                    print("____________Predicted item_________")
                    print(predicted.item())
                    print("________________probs_________")
                    print(probs)           
                    print("________________prob_________")
                    print(prob)   
                    print("________________prob.item_________")
                    print(prob.item())  
                    # if probability == 0:
                    #         print("____________Entered 0 zone____________")
                    #         try:
                    #             wolfram_ssl(result) 
                    #             append_data('data.csv',result,"wolfram")
                    #         except Exception as e:
                    #             print("Exception: ", e)
                    #             InputExecution('google', result)
                    #             append_data('data.csv',result, 'google')
                    #             quick_reply()
                    #         # f = False
                    #         # print("____________changed f to false_______")
                                    
                     
                    if prob.item()>= 0.75:
                        print("____________Entered 0.8 zone____________")
                        for intent in intents ['intents']:
                            if tag == "Bye" and intent["tag"] == "Bye":
                                        reply = random.choice(intent["responses"])  
                                        speak(reply)
                                        append_data('data.csv',result, reply)
                                        exit(0)
                            elif tag == "repeat" and intent["tag"] == "repeat":
                                        prev_response('response')
                                        append_data('data.csv',result,prev_response())
                            elif tag == intent["tag"]:
                                    reply = random.choice(intent["responses"])
                                    append_data('data.csv',result, reply)
                                    print("__________________Final_reply_______________")
                                    print(reply)
                                
                                    if "time" in reply:
                                        NoninputExecution(reply)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    
                                        
                                    elif "date" in reply:
                                        NoninputExecution(reply)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    
                                        
                                    elif "day" in reply:
                                        NoninputExecution(reply)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    
                                    elif "news" in reply:
                                        NoninputExecution(reply)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "joke" in reply:
                                        NoninputExecution(reply)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    
                                    elif "alarm" in reply:
                                        NoninputExecution(reply)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "wait" in reply:
                                        # append_data('data.csv',result, reply)
                                        status = False
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "wikipedia" in reply:
                                        InputExecution(reply, result)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "google" in reply:
                                        InputExecution(reply, result)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "weather" in reply:
                                        InputExecution(reply, result)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "location" in reply:
                                        # append_data('data.csv',result, reply)
                                        InputExecution(reply, result)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "playmusic" in reply:
                                        InputExecution(reply, result)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    else: 
                                        speak(reply)
                                        #quick_reply()
                    elif prob >=0.5 and prob<0.8:
                            print("____________Entered 0.2 zone____________")
                            for intent in intents ['intents']: 
                                if tag == intent["tag"]:
                                    reply = random.choice(intent["responses"])
                                    append_data('data.csv',result, reply)
                                    print("__________________Final_reply_______________")
                                    print(reply)
                                    if "wikipedia" in reply:
                                        InputExecution(reply, result)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "google" in reply:
                                        InputExecution(reply, result)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "weather" in reply:
                                        InputExecution(reply, result)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "location" in reply:
                                        # append_data('data.csv',result, reply)
                                        InputExecution(reply, result)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    elif "playmusic" in reply:
                                        InputExecution(reply, result)
                                        # append_data('data.csv',result, reply)
                                        #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                                    else:
                                        try:
                                            wolfram_ssl(result)
                                            # append_data('data.csv',result, reply)
                                        except Exception as e:
                                            print("Exception: ", e)
                                            InputExecution(reply, result)
                                            # append_data('data.csv',result, reply)
                                            #quick_reply()
                                        # f = False
                                        # print("____________changed f to false_______")
                    else:
                            try:
                                    wolfram_ssl(result)
                                    append_data('data.csv',result,"wolfram")
                            except Exception as e:
                                        print("Exception: ", e)
                                        pywhatkit.search(sentence)
                                        append_data('data.csv',sentence, "finally_googled")
                    status = inactive()    
                        
                                         
        

