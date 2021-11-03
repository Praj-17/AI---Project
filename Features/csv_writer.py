#create and read from a csv file
# import csv 
import datetime
import pandas as pd
time = datetime.datetime.now().strftime("%H: %M")
date = datetime.date.today()
# def get_length(file):
#     return 1
def append_data(file, query, response):
    # fieldnames = ['date', 'time', 'query', 'response']
    #the number of rows?
    # next_id = get_length(file)
    # output = pd.read_csv(file)
    # pd.read_csv(file, delimiter= ',')
    df = pd.DataFrame()
    df = df.append([[date, time,query, response]])
    df.to_csv(file, header= False, mode= "a")
        # writer.writeheader()
        # writer.writerow()

