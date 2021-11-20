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
stemar = PorterStemmer()
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
    return stemar.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Packing all those descrete packed words into some model understandable and sending forward
    
    There is one diadvantage of bag of words though,
    it assigns the same wietage to all the words
    to solve this problem we have another system called
    word2vec or TFIDF
    (term frequency–inverse document frequency)
    """
    sentence_word = [lemmatize(word) for word in tokenized_sentence if not word in set(stopwords.words('english'))]
    bag = np.zeros(len(words), dtype = np.float32)
    
    for index, w in enumerate (words):
        if w in sentence_word:
            bag[index] = 1
    return bag

def bag_of_words_2(tokenized_sentence):
    """
    Packing all those descrete packed words into some model understandable and sending forward
    
    There is one diadvantage of bag of words though,
    it assigns the same wietage to all the words
    to solve this problem we have another system called
    word2vec or TFIDF
    (term frequency–inverse document frequency)
    """
    sentence_word = [lemmatize(word) for word in tokenized_sentence if not word in set(stopwords.words('english')) ]
    bag = cv.fit_transform(sentence_word).toarray()
    return bag
"""
TFIDF-  (term frequency–inverse document frequency)
term Frequency =( no of words in sentence )/no of words in sentence
inverse document frequency==
                          log((no of sentences)/(no of sentences containing words))
________________________
                         |
#Finally = TF*IDF        |
_________________________|
"""
def Tf_Idf(tokenized_sentence):
    sentence_word = [lemmatize(word) for word in tokenized_sentence if not word in set(stopwords.words('english')) ]
    bag = tf_idf.fit_transform(sentence_word)
    return bag

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

