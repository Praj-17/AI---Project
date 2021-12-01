import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Neural_Network.text_preprocessing import ignore_words, tokenize, stem, bag_of_words
from Neural_Network.brain import NeuralNet

## Unwrappping the Json file
with open('intents.json', 'r') as f:
    intents = json.load(f) #Naming the loaded json file as intents which actually is a dictionary 

all_words = [] #it has all the words from the patterns section in the json file
tags = [] #list of all tags present into json file
tag_per_word = [] #All words and tags combine to make tag_per_word

#This for loop will append all tags,patterns from the json file to the tags list
for intent in intents['intents']: #refering to the intents key which has an list of dictinonaries in it called as an intent
     tag = intent['tag']
     tags.append((tag))
     
     for pattern in intent['patterns']:
         # print(pattern)
         w = tokenize(pattern) #breaking all the patterns(in the json) into words and creating a list of words
         #_______________________words lematized instaed stem_____________
         w = [stem(w1) for w1 in w ]
         all_words.extend(w) # adding that list of words to the all_words list
         # all_words = [lemmatize(w) for w in all_words if w not in ignore_words]
         tag_per_word.append((w,tag)) #Collecting all words and their respective tags at one place
         
     
     
all_words = sorted(set(all_words))
tags = sorted(set(tags))
 #______________________converting lists to numpy arrays for speed
x_train = []  #it will store the actual tags
y_train = []  # it will store their respective indices
for(pattern_sentence, tag) in tag_per_word:
    print(pattern_sentence)
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
     
    label = list(tags).index(tag) #training it on the index of the tag
    y_train.append(label)


if __name__ == '__main__':
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
        def __init__(self,x_train,y_train) :
            self.n_samples = len(x_train)
            self.x_data = x_train
            self.y_data = y_train
        def __getitem__(self,index):
            return self.x_data [index], self.y_data[index]
        def __len__(self):
            return self.n_samples
        
    dataset = ChatDataset(x_train,y_train)
    train_loader = DataLoader(dataset= dataset, batch_size = batch_size , shuffle = True, num_workers = 0)

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
