import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Neural_Network.text_preprocessing import tokenize, ignore_words,Tf_Idf, bag_of_words,preprocessing
from Neural_Network.brain import NeuralNet

## Unwrappping the Json file
try:
    with open('intents.json', 'r') as f:
        intents = json.load(f) #Naming the loaded json file as intents which actually is a dictionary 
except Exception as e:
    print(e)

all_patterns = [] 
tags = [] 
tag_per_sentnece = [] 
trained_words = []
for intent in intents['intents']: 
     tag = intent['tag']
     tags.append((tag))     
     for pattern in intent['patterns']:
        processed_patterns = preprocessing(pattern)
        all_patterns.append((pattern))
        tag_per_sentnece.append((pattern, tag))
     
all_patterns = sorted(set(all_patterns))
tags = sorted(set(tags))
x_train = []  #it will store the actual tags
y_train = []  # it will store their respective indices
bag = Tf_Idf(all_patterns)
x_train.append(bag)
for(pattern_sentence, tag) in tag_per_sentnece:
    if pattern_sentence != [] :
        trained_words.extend(pattern_sentence)
        label = list(tags).index(tag) #training it on the index of the tag
        y_train.append(label)
        
x_train = np.array(x_train)
y_train = np.array(y_train)
if __name__ == '__main__':
    x_train = np.array(x_train)
    y_train = np.array(y_train)
   
    #Specifying the feautres
    num_epochs = 1000
    batch_size = 8
    # batch_size =1
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
        "all_patterns" : all_patterns,
        "tags" : tags
    }

    FILE = "TrainData.pth" #pth files train models very faster than other models 
    torch.save(data, FILE)

    print((f"Training Complete, File saved to {FILE}"))

