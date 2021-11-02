import numpy as np 
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
# nltk.download('punkt')


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
def stem(word):
    """
    It will add some AIness to the code 
    That is, 
    it will consider the words like [hello, hi, hey] in the same way
    eg - it will convert
    ["Hello", "Jarvis"] to [[h,e,l,l,o], [j,a,r,v,i,s]]
    
    """
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, words):
    """
    Packing all those descrete packed words into some model understandable and sending forward
    """
    sentence_word = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype = np.float32)
    
    for idx, w in enumerate (words):
        if w in sentence_word:
            bag[idx] = 1
    return bag
    