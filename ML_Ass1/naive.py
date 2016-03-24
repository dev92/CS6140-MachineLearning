
__author__ = 'Dev'

''' The libraries used for this code are as follows:
    1) Sci-kit learn => Naives Bayes model and to use label encoder
    2) Pandas => To form Data Matrix
'''

import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB


# Function to calculate Accuracy and print the stats

def getAccuracy(predicted,actual):

    error = 0
    correct = 0

    test_samples = len(actual)

    for x,y in zip(predicted,actual):
        if x!=y:
            error+=1
        else:
            correct+=1

    print("Total Test Data: {}\nCorrect Predictions: {}\nWrong Predictions: {}\n"
          "Accuracy: {}%".format(test_samples,correct,error, round(correct/float(test_samples)*100,3)))



def trainandPredict(trainset,testset):

    # Extracting features except last column which is the labels

    features = trainset[colnames[:len(colnames)-1]].values
    ylabel = trainset['label'].values

    # forming Test features
    test_features = testset[colnames[:len(colnames)-1]].values

    gnb = GaussianNB()

    # Training a Gaussian Model and predicting on Test data
    y_pred = gnb.fit(features, ylabel).predict(test_features)

    # function to get statistics on the predicted values
    getAccuracy(y_pred,testset['label'].values)




if __name__ == '__main__':


    #Column Names for forming DataFrame

    colnames = ['age','workclass','fnlwgt','education','education-num',
                'marital-status','occupation','relationship','race','sex',
                'capital-gain','capital-loss','hours-per-week',
                'native-country','label']


    # Columns which have categorical data

    categoricalcols = ['workclass','education','marital-status',
                       'occupation','relationship','race','sex',
                       'native-country','label']


    # Reading Train data and Test data from files, replacing NaN Values with "?" to maintain uniformity

    trainset = pd.read_csv("census/adult.data",keep_default_na=False, na_values= "?",skipinitialspace=True,names=colnames,header=None)

    testset = pd.read_csv("census/adult.test",keep_default_na=False, na_values= "?",skipinitialspace=True,names=colnames,header=None)


    # Using Label encoder from Sci-kit to convert categorical data into numbers
    # it also takes care of replacing missing with an appropriate code
    le = preprocessing.LabelEncoder()


    # For each categorical column, transforming categories into appropriate codes (int)
    for name in categoricalcols:
        le.fit(trainset[name].str.strip('.'))
        trainset[name] = le.transform(trainset[name].str.strip('.'))
        testset[name] = le.transform(testset[name].str.strip('.'))


    # Function to Train and Test model
    trainandPredict(trainset,testset)






