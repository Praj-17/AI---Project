import random
import json
from requests.models import Response

from torch.nn.modules import module
from brain import NeuralNet
from text_preprocessing import bag_of_words, tokenize
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)


FILE = "TrainData.pth"
data  = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval

#---------------

Name = "Jarvis"
from listen import listen
from speak import speak
from task import InputExecution, NoninputExecution
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
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim = 1)
    prob = probs[0][predicted.item()]
    if prob.item()> 0.75:
        for intent in intents ['intents']:
                if tag == "Bye" and intent["tag"] == "Bye":  
                    speak(random.choice(intent["responses"]))
                    exit(0)
    if prob.item()> 0.75:
        for intent in intents ['intents']:
            if tag == intent["tag"]:
                reply = random.choice(intent["responses"])
                
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
                else:
                    speak(reply)
                
                
                    
while True:
     main()
        
    
