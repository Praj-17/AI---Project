import pandas as pd

data = pd.read_csv('data.csv')
# print (data)
print(data.iloc[-1:])
# print(data.loc[:-5])
# print(data.tail())
print(data.iloc[-1:]['Date'])