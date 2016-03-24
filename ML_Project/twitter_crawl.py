import re

from pattern.web import Twitter,hashtags,author
from pattern.db  import Datasheet, pprint, pd
from pattern.web import Google, plaintext


table = Datasheet()
index = set()
texts = set()


table.append(['Tweets'])


'''
Cleaning the tweets extracted from a particular user timeline,
code is available in another file
'''


with open('extracted_tweets_translated.txt') as f:
    for tweet in f.readlines():
        clean_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
        print

        print clean_text
        print author(tweet)
        print hashtags(tweet)

        print

        table.append([clean_text])



'''
Keyword based crawling
'''


twitter = Twitter(language='en')


for word in ["deaththreat","attack","murder","riot","crime","kill","angry","protest","boycott","torture"]:

    prev = None

    print "processing word:",word

    for tweet in twitter.search(word,start=prev,cached=False,count=200):

        # print
        #
        # print tweet.text
        # print tweet.author
        # print tweet.date
        # print hashtags(tweet.text)
        #
        # print
        clean_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet.text).split())

        if tweet.id not in index and clean_text not in texts:
            table.append([tweet.id,tweet.text,clean_text,hashtags(tweet.txt)])
            index.add(tweet.id)
            texts.add(clean_text)

        prev = tweet.id
#
table.save(pd("tweets_threats.csv"))


# pprint(table,truncate=100)


