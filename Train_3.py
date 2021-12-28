# importing modules
import pandas as pd
import json
from Neural_Network.text_preprocessing import preprocessing, ignore_words
import warnings
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
warnings.filterwarnings("ignore")
def get_data(file):
    try:
        with open(file, 'r') as f:
            intents = json.load(f) #Naming the loaded json file as intents which actually is a dictionary 
    except Exception as e:
        print(e)

    all_patterns = [] 
    tags = [] 
    tag_per_sentence = [] 


    for intent in intents['intents']: 
        tag = intent['tag']
        tags.append((tag))     
        for pattern in intent['patterns']:
            processed_pattern = preprocessing(pattern)
            tag_per_sentence.append((processed_pattern, tag))
    data = pd.DataFrame(tag_per_sentence,columns =['Patterns','tags'])
    data.drop_duplicates()
    return data

data = get_data('intents_1.json')
def prepare_data():
    data['intent_id'] = data['tags'].factorize()[0]
    intent_id_df = data[['tags', 'intent_id']].drop_duplicates().sort_values('intent_id')
    intent_to_id = dict(intent_id_df.values)
    id_to_intent = dict(intent_id_df[['intent_id', 'tags']].values) #This will add the column in our dataframe
    return data
def naive_algo():
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=ignore_words)
    df=prepare_data()
    features = tfidf.fit_transform(df.Patterns).toarray()
    labels = df.intent_id
    features.shape
    X_train, X_test, y_train, y_test = train_test_split(df['Patterns'], df['tags'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    return clf,count_vect
naive_algo()
def predict(query):
    clf,count_vect=naive_algo()
    intent=clf.predict(count_vect.transform([preprocessing(query)]))
    print(intent)
    intent=str(intent).strip("['']")
    return intent
print(predict("what an"))