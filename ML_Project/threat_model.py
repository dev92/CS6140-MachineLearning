import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix
from tabulate import tabulate
import matplotlib.pyplot as plt
import argparse


def msg(name=None):
    return '''threat_model.py
         [-t, Pass argument for type of model
         [-t 1 for selective words as features]

         if no argument is passed then it uses all words

         example usage :  python threat_model.py  -t 1
         example usage :  python threat_model.py


        '''


def tweet_to_words( raw_tweet ):
    # Function to convert a raw tweet to a string of words
    # The input is a single string (a raw tweet), and
    # the output is a single string (a preprocessed tweet)


    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_tweet)


    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()


    # 3. Remove stop words
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]

    # 4. Joining the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Primal Perceptron algorithm",usage=msg())
    parser.add_argument("-t","--type",type = int,required = False)


    args = parser.parse_args()
    model_type = args.type

    trainset = pd.read_csv('FinalTrain_12000.csv',keep_default_na=False)

    trainset = trainset.drop('word_count', 1)



    selective_words = ["hate","kill","exterminate","dogs","protest","protesting","demonstration","demonstrate","yell","scream","cunt","jib","shit","fuck",\
          "fight","stink","parade","marching","calling","riot","rioting","police","hassle","out of control","angry","take over",\
          "argument","attacked","attacking","fighting","racing","fought","beaten","injured","wounded","destruction","vandalism","climbing",\
          "damage","fire","stabbing","blows","arrests","alarm","perpetrator","harness","rude","anti","healing","agent","abuse","detection",\
          "violence","threat","intimidation","death threats","death","murder","fraud","incident","growing resistance","retarded","come all",\
          " events","attacks","negativity","irritating","war","over-stimulated"]


    clean_train_tweets = []


    for i in xrange( 0, len(trainset) ):
        if( (i+1)%1000 == 0 ):
            print "%d Tweets cleaned" % (i+1)
        clean_train_tweets.append( tweet_to_words(trainset["Tweets"][i]) )


    if model_type == 1:
        print "\nYou have chosen selective words as features\n"
        roc_type = "Selective words as features"
        vocabulary = selective_words
    else:
        roc_type = "With bag of words as features"
        vocabulary  = None

    vectorizer = CountVectorizer(stop_words = ['b','rt','death','threat'],vocabulary=vocabulary)


    features = vectorizer.fit_transform(clean_train_tweets)


    train_set, test_set, train_labels, test_labels = train_test_split(features.toarray(),trainset['label'],
                                                                      test_size=0.2,random_state=42)

    test_labels = test_labels.reshape(len(test_labels),1)

    gnb = MultinomialNB()

    # Training a Gaussian Model and predicting on Test data
    y_pred = gnb.fit(train_set, train_labels).predict(test_set).reshape(len(test_labels),1)




    fpr, tpr , _ = roc_curve(test_labels[:,0],y_pred[:,0])

    roc_auc = auc(fpr,tpr)




    c_matrix = confusion_matrix(test_labels, y_pred,labels=[0,1])

    table = [["Actual:\nNo threat\n"], ["Actual:\nThreat"]]

    table[0].extend(list(c_matrix[0]))

    table[1].extend(list(c_matrix[1]))

    print "\n\t\t\t\tConfusion Matrix\n"

    print tabulate(table,headers=["Predicted: No threat","Predicted: Threat"])



    print("\n\t\t\tTotal Test Data: {}\n\t\t\tAccuracy: {}%".format(len(test_labels),round(accuracy_score(test_labels,y_pred) * 100,3)))

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.10])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: '+roc_type)
    plt.legend(loc="lower right")
    plt.show()




