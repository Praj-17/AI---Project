import requests
import json



def get_news():
    url = 'https://newsapi.org/v2/top-headlines?country=in&apiKey=8f6bbf4ba75a402fbfbe8e12704272e2'
    news = requests.get(url).text
    news_dict = json.loads(news)
    articles = news_dict['articles']
    try:

        return articles
    except:
        return False


def getNewsUrl():
    return 'https://newsapi.org/v2/top-headlines?country=in&apiKey=8f6bbf4ba75a402fbfbe8e12704272e2'
