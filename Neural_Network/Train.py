from pickle import TRUE
from textwrap import indent
import numpy as np
import json
import torch
from torch.autograd.grad_mode import F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from brain import  NeuralNet
from text_preprocessing import bag_of_words, tokenize, stem
## Unwrappping the Json file
with open('intents.json', 'r') as f:
    intents = json.load(f) #Naming the loaded json file as intents which actually is a dictionary 

all_words = []
tags = [] # these the  predifened tags that we have created in the json file
xy = [] #All words and tags combine to make xy

#This for loop will append all tags,patterns from the json file to the tags list
for intent in intents['intents']:
    tag = intent['tag']
    tags.append((tag))
    
    for pattern in intent['patterns']:
        # print(pattern)
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
ignore_words = [",", ".", "?", "/", "!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []  #it will store the actual tags
y_train = []  # it will store their respective indices

for(pattern_sentence, tag) in xy:
    
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)
    
x_train = np.array(x_train)
y_train = np.array(y_train)


#Specifying the feautres
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

print("Training the Model")

class ChatDataset(Dataset):
    def __init__(self) :
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples
    
dataset = ChatDataset()
train_loader = DataLoader(dataset= dataset, batch_size = batch_size , shuffle = TRUE, num_workers = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype = torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1)% 100 == 0:
            print(f"Epoch[{(epoch+1)}/ {num_epochs}], loss: {loss.item(): .4f}")
print(f"Final loss: {loss.item(): .4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words" : all_words,
    "tags" : tags
}

FILE = "TrainData.pth" #pth files train models very faster than other models 
torch.save(data, FILE)

print((f"Training Complete, File saved to {FILE}"))
