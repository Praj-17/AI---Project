#create and read from a csv file
# import csv 
import datetime
import pandas as pd
time = datetime.datetime.now().strftime("%H: %M")
date = datetime.date.today()

# def get_length(file):
#     return 1
def append_data(file , query, response):
    # fieldnames = ['date', 'time', 'query', 'response']
    #the number of rows?
    # next_id = get_length(file)
    # output = pd.read_csv(file)
    # sr_no = 0 
    pd.read_csv(file, delimiter= ',')
    df = pd.DataFrame()
    df = df.append([[date, time,query, response]]).set_index(0.00,drop=True)
    df.to_csv(file ,header= False, mode= "a")
    # df.add()
    # sr_no += 1 
    # output =pd.read_csv('data.csv')
    # print(output)
    # output.append(pd.DataFrame({'Date': date, 'time':time, 'query':query, 'response': response}, ignore_index = True))
    # print(output.head())
        # writer.writeheader()
        # writer.writerow()
