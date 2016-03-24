from __future__ import division

import argparse
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd



def msg(name=None):
    return '''Dual_perceptron.py
         [-k 1, Kernel type 1 for LinearKernel]
         [-k 2, Kernel type 1 for GaussianKernel]
         [-f, Pass argument for filepath]

         example usage :  python Dual_perceptron.py -k 2 -f a2_datasets/perceptron/percep2.txt

        '''


'''
Linear Kernel function which takes dot product between each training sample and entire training examples
multiplied with alphas and training labels
'''

def linearkernelfunc(i):
    product_term = np.multiply(alphas,trainlabels)
    result = np.dot(training_examples.ix[i],np.array(training_examples).transpose())
    result = np.dot(product_term,result)
    return result + bias

'''
Gaussian Kernel function which takes square euclidean distance between each training sample and entire training examples
multiplied with alphas and training labels.
The euclidean distance matrix is precomputed for faster running of program
'''

def gaussianKernel(i):
    result = np.multiply(alphas,trainlabels)
    result = np.dot(result,np.array(euclideandistances[i].transpose()))
    return result+bias

'''
Function to determine which kernel function to be used
'''

def Kernelfunc(type,i):
    if type == 1:
        return linearkernelfunc(i)
    elif type == 2:
        return gaussianKernel(i)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Dual Perceptron algorithm with Kernel functions",usage=msg())
    parser.add_argument("-k","--Kernel",type = int,required = True)
    parser.add_argument("-f","--filepath",type = str,required = True)


    args = parser.parse_args()

    Kerneltype = args.Kernel

    filepath = args.filepath

    '''
    Reading input file and extracting features and labels
    '''

    trainset = pd.read_csv(filepath,delim_whitespace=True,header=None)

    num_of_examples = trainset.shape[0]

    num_of_features = trainset.shape[1]-1

    alphas = np.zeros(num_of_examples)

    bias = 0.0


    trainlabels = trainset[trainset.columns[num_of_features]]


    training_examples = trainset[trainset.columns[:num_of_features]]

    '''
    Precomputing Gaussian kernel values for every pair in the training set
    '''

    if Kerneltype == 2:
        euclideandistances = np.exp(np.multiply(-0.25,np.square(cdist(training_examples,training_examples,'euclidean'))))



    iterations = 0

    '''
    Algorithm starts here, it loops till convergence condition is met,
    condition is when no mistakes occur
    '''


    while True:
        iterations+=1
        incorrect = 0
        for i in range(0,num_of_examples):
            label = trainlabels[i]
            if label*Kernelfunc(Kerneltype,i) <=0:
                incorrect+=1
                alphas[i]+=1
                bias = bias + label
        print("Iteration:{} Incorrect:{}".format(iterations ,incorrect))
        if incorrect == 0:
            break



    correct = 0

    print "Bias:",bias

    '''
    Using learnt alphas we are trying to recontruct the labels to verify if predictions is same
    '''


    for i in range(0,num_of_examples):
        f = Kernelfunc(Kerneltype,i)
        if f >=0 and trainlabels[i] == 1:
            correct+=1
        elif f < 0 and trainlabels[i] == -1:
            correct+=1

    print("\nTotal Data: {}\nCorrect Predictions: {}\n"
          "Accuracy: {}%".format(num_of_examples,correct,round((correct/num_of_examples)*100,3)))





