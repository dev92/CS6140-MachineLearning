from __future__ import division
import json
import os.path
from datetime import datetime


def convertJSON(trainfile):

    train_dict={}


    train_dict["_RARE_"] = {}
    train_dict["unigram"] = {}
    train_dict["bigram"] = {}
    train_dict["trigram"] = {}

    # print train_dict

    with open(trainfile,mode='r') as f:
        for line in f.readlines():
            curr = line.split()
            count = int(curr[0])

            if curr[1].startswith("WORDTAG"):

                if curr[3].strip() not in train_dict:
                    train_dict[curr[3].strip()] = {curr[2].strip():count}

                else:
                    train_dict[curr[3].strip()].update({curr[2].strip():count})

                if curr[2].strip() in train_dict["_RARE_"] and count < 5:
                    train_dict["_RARE_"][curr[2].strip()]+=count

                elif count < 5:
                    train_dict["_RARE_"].update({curr[2].strip():count})

            elif curr[1].startswith("1-GRAM"):
                train_dict["unigram"][curr[2].strip()] = count

            elif curr[1].startswith("2-GRAM"):
                n_gram = " ".join(curr[2:]).strip()
                train_dict["bigram"][n_gram] = count

            elif curr[1].startswith("3-GRAM"):
                n_gram = " ".join(curr[2:]).strip()
                train_dict["trigram"][n_gram] = count


    # print train_dict["unigram"]


    with open("train_counts.json", 'w') as j:
        json.dump(train_dict,j,ensure_ascii=False)

    print "File Generated for future use!"


def readTestfile(testfile):

    test_sentences = []
    sentence = []
    with open(testfile) as test:
        for word in test.readlines():
            if word.strip():
                sentence.append(word.strip())
            else:
                test_sentences.append(" ".join(sentence))
                sentence = []

    return test_sentences

def calculateQ(w,u,v):


    try:
        trigram = trainData["trigram"][" ".join([w,u,v])]
        # print " ".join([w,u,v]) , trigram
    except KeyError:
        trigram = 0.0
    try:
        bigram = trainData["bigram"][" ".join([w,u])]
        # print " ".join([w,u]) , bigram
    except KeyError:
        bigram = 0.0

    try:
        return trigram/bigram
    except ZeroDivisionError:
        return 0.0

def calculateE(word,v):
    if word in trainData:
        try:
            return trainData[word][v]/trainData["unigram"][v]
        except KeyError:
            # print word,v
            return 0.0
            # return trainData["_RARE_"][v]/trainData["unigram"][v]

    else:
        # print "Word doesnt exist!"
        return trainData["_RARE_"][v]/trainData["unigram"][v]


def get_tags(i):
    if i == 0 or i == -1:
        return ['*']
    else:
        return tags


def viterbiAlgorithm(sentence):


    pi = {}
    bp = {}

    pi[0,'*','*'] = 1
    bp['*','*'] = []

    words = sentence.split()

    for k in xrange(1,len(words)+1):

        temp_path = {}

        for u in get_tags(k-1):
            for v in get_tags(k):
                pi[k,u,v],prev_w = max([(pi[k-1,w,u] * calculateQ(w,u,v) * calculateE(words[k-1],v),w) for w in get_tags(k-2)])
                temp_path[u,v] = bp[prev_w,u] + [v]

        bp = temp_path
        # print bp


    prob,umax,vmax = max([(pi[len(words),u,v] * calculateQ(u,v,'STOP'),u,v) for u in get_tags(len(words)-1) for v in tags])


    for word,tag in zip(words,bp[umax,vmax]):
        # print word,tag
        output.write(word+" "+tag+"\n")




    '''
    :param sentence:
    :return: None
    '''
    '''
    Alternate technique for faster processing, but getting 82.8% accuracy
    reason: avoiding few combination of tags
    '''
    # pi_probabilities = []
    #
    # pi_initial = 1
    # u = '*'
    # w = '*'
    #
    # for word in sentence.split():
    #     # print word
    #
    #     for tag in tags:
    #         # print tag
    #         emission = calculateE(word,tag)
    #         q = calculateQ(w,u,tag)
    #         # print q,emission,pi_initial
    #         pi_probabilities.append(pi_initial * q * emission)
    #
    #     # print pi_probabilities
    #     max_prob = max(pi_probabilities)
    #     # print max_prob
    #     index_v = pi_probabilities.index(max_prob)
    #     # print word,tags[index_v]
    #     line = word+" "+tags[index_v]+"\n"
    #     output.write(line)
    #     pi_initial = max_prob
    #     pi_probabilities = []
    #     w = u
    #     u = tags[index_v]



if __name__ == '__main__':

    if not os.path.isfile("train_counts.json"):
        print "Converting train.counts into json format for ease of access...."
        convertJSON('a3_datasets/UD_English/train.counts')

    if os.path.isfile("predicted_tags.txt"):
        os.remove("predicted_tags.txt")


    with open("train_counts.json",'r') as f:
        trainData  = json.load(f)


    testSentences = readTestfile('a3_datasets/UD_English/test.words')

    output = open("predicted_tags.txt",mode='w')

    tags = trainData["unigram"].keys()

    start_time = datetime.now()

    for sentence in testSentences:
        viterbiAlgorithm(sentence)
        output.write("\n")

    output.close()

    print "Running time:",(datetime.now()-start_time)













