import random
import json
from requests.models import Response
from torch.nn.modules import module
import wolframalpha
from Neural_Network.brain import NeuralNet
from Neural_Network.text_preprocessing import bag_of_words, tokenize
import torch
from Features.csv_writer import append_data
from task import prev_response
from Features.wishme import wishMe
from Features.wolfram import wolfram_ssl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)


FILE = "TrainData.pth"
data  = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]

model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval

#---------------

from Features.listen import listen
from Features.speak import speak
from task import InputExecution, NoninputExecution, read_prev_response
def main():
    sentence =listen()
    result = str(sentence)
    
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1,x.shape[0])
    x = torch.from_numpy(x).to(device)
    
    output = model(x)
    _, predicted = torch.max(output, dim = 1 )
    # print(predicted)
    tags = data["tags"]
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim = 1)
    prob = probs[0][predicted.item()]
    
    # if prob.item()> 0.75:
    #     for intent in intents ['intents']:
                    
    if prob.item()> 0.8:
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
               
                if "time" in reply:
                    NoninputExecution(reply)
                elif "date" in reply:
                    NoninputExecution(reply)
                elif "day" in reply:
                    NoninputExecution(reply)
                elif "wikipedia" in reply:
                    InputExecution(reply, result)
                elif "google" in reply:
                    InputExecution(reply, result)
                elif "news" in reply:
                    NoninputExecution(reply)
                elif "joke" in reply:
                    NoninputExecution(reply)
                elif "weather" in reply:
                    InputExecution(reply, result)
                elif "alarm" in reply:
                    NoninputExecution(reply)
                elif "playmusic" in reply:
                    InputExecution(reply, result)
                elif "wait" in reply:
                    NoninputExecution(reply)
                elif "location" in reply:
                    InputExecution(reply, result)
                    
                elif reply == "Couldn't understand, say that again please!":
                    try:
                        wolfram_ssl()
                    except:  speak("I'm sorry , I don't know that.")
                else: speak(reply)
    else:
        append_data('data.csv',result, "Couldn't understand, say that again please!")            
                
if __name__ == "__main__":    
    wishMe()          
    while True :    
        main()
        
    
