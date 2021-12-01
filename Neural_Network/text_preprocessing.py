from nltk.stem import porter
import numpy as np 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# import re
lemmatizer = WordNetLemmatizer()
cv = CountVectorizer()
tf_idf = TfidfVectorizer()
# nltk.download('punkt')
# nltk.download('wordnet')

"""
Adding lemmitrizer will help improve the performance
"""
# query = "hello", "Hellos", "hello?"
# thi is how tokenization happens
# h,e,l,l,o

# H,e,l,l,o ,s

# h,e,l,l,o ,?
#consider
# Hello Jarvis - is the input it h

#______________________ignore words ready___________
ignore_words = [',', '.', '?', '/', '!', 'm','s', 'u', 'I','nt', 'can', 'help', 'please','assistant','assist','do', 'does','did','have','has','had','he','she','it','might', 'may','must','me', 'i', 'myself', 'tell', 'say', 'is', 'are', 'a','an','the'
                ]    #removing punctuations
# ignore_words.extend(list(stopwords.words('english')))
# remove_from_stopwords= ['what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'how', 'you', 'u']
# for i in remove_from_stopwords:
#      ignore_words.remove(i),  #only adding stopwords of english lang

def tokenize(sentence):
    """
    It will split the query into multiple characters using NLP
    eg - it will convert Hello Jarvis to 
    ["Hello", "Jarvis"]
    """
    return nltk.word_tokenize(sentence)


def lemmatize(word):
    """
    It will take the word and covert it to its meanigfull format
    finally  final finale finalized ---> 
    """
    return lemmatizer.lemmatize(word)

def stem(word):
    """
    It will add some AIness to the code 
    That is, 
    it will consider the words like [hello, hi, hey] in the same way
    eg - it will convert
    ["Final ", "Finalized", "Finally", "finale"]
    to [Final] because it is common in all the words, rest all is considered as the suffix and is exploited
    
    """
    return PorterStemmer().stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Packing all those descrete packed words into some model understandable and sending forward
    
    There is one diadvantage of bag of words though,
    it assigns the same wietage to all the words
    to solve this problem we have another system called
    word2vec or TFIDF
    (term frequency–inverse document frequency)
    """
    tokenized_sentence = [stem(word) for word in tokenized_sentence if word not in ignore_words]
    bag = np.zeros(len(words), dtype = np.float32) #creating an array of all zeros 
    
    for index, w in enumerate (words):
        if w in tokenized_sentence:
            bag[index] = 1
    return bag

"""
Here i am attempting to figure out a better way rather than 
going for bag of words
Some ways that i have written

1. CountVectorizer - 
2. TFIDF

"""

def bag_of_words_2(tokenized_sentence):
    """
    Packing all those descrete packed words into some model understandable and sending forward
    
    
    There is one diadvantage of bag of words though,
    it assigns the same wietage to all the words
    to solve this problem we have another system called
    word2vec or TFIDF
    (term frequency–inverse document frequency)
    """
    sentence_word = [lemmatize(word) for word in tokenized_sentence if not word in set(ignore_words)]
    bag = cv.fit_transform(sentence_word).toarray()
    return bag
"""
TFIDF-  (term frequency–inverse document frequency)
term Frequency =( no of words in sentence )/no of words in sentence
inverse document frequency==
                          log((no of sentences)/(no of sentences containing words))
__________________________
                         |
#Finally = TF*IDF        |
_________________________|
"""
def Tf_Idf(tokenized_sentence, words):
    sentence_word = [lemmatize(word) for word in tokenized_sentence if not word in set(stopwords.words('english')) ]
    bag = tf_idf.fit_transform([sentence_word]).toarray()
    for index, w in enumerate (words):
        if w in sentence_word:
            bag[index] = 1
    return bag
"""
                  f1     f2       f3
                boy    good    girl
boy is good       1       1       0
girl is good      0       1       1
boy girl good     1       1       1

"""

# "trying out some examples"
# list1 = ['kites', 'babies', 'dogs', 'flying', 'smiling',
#          'driving', 'died', 'tried', 'feet']
# for words in list1:
#     print(f"{words} --->{lemmatize(words)}")
# print(lemmatize('feet'))
# print(stem('feet'))


# sentences = ' '.join(words)

# re.sub('[^a-zA-Z]', ' ') it will remove all the punctuation marks
# print(stopwords.words('english'))

