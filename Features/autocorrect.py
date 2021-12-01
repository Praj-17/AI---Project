from textblob import TextBlob
s = "he is a grat person"
print("_______original_text______")
print(s)
b = TextBlob(s)
print("________Corrected text_____")
print( str(TextBlob(s).correct()))

