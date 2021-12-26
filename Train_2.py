# importing modules
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model
import json
from Neural_Network.text_preprocessing import ignore_words,preprocessing
from similar_words import Get_Similarities
import warnings
warnings.filterwarnings("ignore")
try:
    with open('intents.json', 'r') as f:
        intents = json.load(f) #Naming the loaded json file as intents which actually is a dictionary 
except Exception as e:
    print(e)

all_patterns = [] 
tags = [] 
tag_per_sentence = [] 
trained_words = []

for intent in intents['intents']: 
     tag = intent['tag']
     tags.append((tag))     
     for pattern in intent['patterns']:
        processed_patterns = preprocessing(pattern)
        similar_words = Get_Similarities(processed_patterns)
        similar_words = similar_words.split()
        for i in similar_words:
            all_patterns.append((str(i).lower() ))
            tag_per_sentence.append((pattern, tag))
data = pd.DataFrame(tag_per_sentence,columns =['Patterns','tags'])
data = data.dropna()
# Here, ngram_range specifies how to transform the data, ngram_range of (1, 2) will have both monograms and bigrams in the Tf-Idf vectors. stop_words specifies the language from which the stop words to be removed.

if __name__ == '__main__':
    vectorizer  = TfidfVectorizer(ngram_range=(1,2), stop_words=ignore_words)
    training_data_tfidf = vectorizer.fit_transform(data['Patterns']).toarray()
    #One - hot encoding
    le = LabelEncoder()
    training_data_tags_le = pd.DataFrame({"tags": le.fit_transform(data['tags'])})
    training_data_tags_dummy_encoded = pd.get_dummies(training_data_tags_le["tags"]).to_numpy()

    chatbot = Sequential()
    chatbot.add(Dense(10, input_shape=(len(training_data_tfidf[0]),)))
    chatbot.add(Dense(8))
    chatbot.add(Dense(8))
    chatbot.add(Dense(6))
    chatbot.add(Dense(len(training_data_tags_dummy_encoded[0]), activation="softmax"))
    chatbot.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # fitting DNN
    chatbot.fit(training_data_tfidf, training_data_tags_dummy_encoded, epochs=500, batch_size=100)
 
    print(chatbot.summary())
    # saving model file
    save_model(chatbot, "TrainData")

    