import random
import json
from Neural_Network.brain import NeuralNet
from Neural_Network.text_preprocessing import stem
from Neural_Network.text_preprocessing import bag_of_words, tokenize,ignore_words
import torch
from Features.csv_writer import append_data
from task import prev_response
from Features.wishme import wishMe
from Features.wolfram import wolfram_ssl
from subprocess import call

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)


FILE = 'TrainData.pth'
try:
    model  = torch.load(FILE)
except Exception as e: 
    print(e)
    call(["Python3", FILE])
    model  = torch.load(FILE)
except:
    print("ok")
model_state = model["model_state"]
all_words = model["all_words"]
tags = model["tags"]
model = NeuralNet(model["input_size"],model["hidden_size"], model["output_size"]).to(device)
model.load_state_dict( model_state)
model.eval

#---------------------

from Features.listen import listen 
from Features.speak import speak
from task import InputExecution, NoninputExecution, read_prev_response
def main():
    sentence =listen()
    result = str(sentence)
    #lets say sentence = "What is photosynthesis"
    
    sentence = tokenize(sentence)
    sentence = [stem(w) for w in sentence if w not in ignore_words]
    print(sentence)
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
    print("________________probs_________")
    print(probs)           
    print("________________prob_________")
    print(prob)           
    if prob.item()>= 0.8:
        for intent in intents ['intents']:
            if tag == "Bye" and intent["tag"] == "Bye":
                    reply = random.choice(intent["responses"])  
                    speak(reply)
                    append_data('data.csv',result, reply)
                    exit(0)
            elif tag == "repeat" and intent["tag"] == "repeat":
                    read_prev_response()
                    append_data('data.csv',result,prev_response())
                    break
            elif tag == intent["tag"]:
                reply = random.choice(intent["responses"])
                append_data('data.csv',result, reply)
                print(reply)
               
                if "time" in reply:
                    NoninputExecution(reply)
                    append_data('data.csv',result, reply)
                elif "date" in reply:
                    NoninputExecution(reply)
                    append_data('data.csv',result, reply)
                elif "day" in reply:
                    NoninputExecution(reply)
                    append_data('data.csv',result, reply)
                
                elif "news" in reply:
                    NoninputExecution(reply)
                    append_data('data.csv',result, reply)
                elif "joke" in reply:
                    NoninputExecution(reply)
                    append_data('data.csv',result, reply)
                
                elif "alarm" in reply:
                    NoninputExecution(reply)
                    append_data('data.csv',result, reply)
                elif "wait" in reply:
                    NoninputExecution(reply)
                    append_data('data.csv',result, reply)
                elif "wikipedia" in reply:
                    InputExecution(reply, result)
                    append_data('data.csv',result, reply)
                elif "google" in reply:
                    InputExecution(reply, result)
                    append_data('data.csv',result, reply)
                elif "weather" in reply:
                    InputExecution(reply, result)
                    append_data('data.csv',result, reply)
                elif "location" in reply:
                    append_data('data.csv',result, reply)
                    InputExecution(reply, result)
                elif "playmusic" in reply:
                    InputExecution(reply, result)
                    append_data('data.csv',result, reply)
                else: speak(reply)
    elif prob >=0.2 and prob<0.8:
        for intent in intents ['intents']: 
            if tag == intent["tag"]:
                reply = random.choice(intent["responses"])
                append_data('data.csv',result, reply)
                print("__________________Final_reply_______________")
                print(reply)
                if "wikipedia" in reply:
                    InputExecution(reply, result)
                    append_data('data.csv',result, reply)
                elif "google" in reply:
                    InputExecution(reply, result)
                    append_data('data.csv',result, reply)
                elif "weather" in reply:
                    InputExecution(reply, result)
                    append_data('data.csv',result, reply)
                elif "location" in reply:
                    append_data('data.csv',result, reply)
                    InputExecution(reply, result)
                elif "playmusic" in reply:
                    InputExecution(reply, result)
                    append_data('data.csv',result, reply)
                else:
                    try:
                        print("entering wolfram 1")
                        wolfram_ssl()
                    except: 
                        print("entered except")
                        speak("I'm sorry , I don't know that.")
                    append_data('data.csv',result, "Couldn't understand, say that again please!")   
    else:
        try:
            print("entering wolfram")
            wolfram_ssl()
        except: 
            print("entered except")
            speak("I'm sorry , I don't know that.")
        append_data('data.csv',result, "Couldn't understand, say that again please!")            
                
if __name__ == "__main__":    
    wishMe()          
    while True :   
        main()
        
    
