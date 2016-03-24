from __future__ import division

import argparse
import numpy as np
import pandas as pd



def msg(name=None):
    return '''Primal_perceptron.py
         [-f, Pass argument for filepath]

         example usage :  python Primal_perceptron.py  -f a2_datasets/perceptron/percep1.txt

        '''


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Primal Perceptron algorithm",usage=msg())
    parser.add_argument("-f","--filepath",type = str,required = True)


    args = parser.parse_args()
    filepath = args.filepath



    '''
    Reading input file and extracting features and labels
    '''


    trainset = pd.read_csv(filepath,delim_whitespace=True,header=None)

    num_of_examples = trainset.shape[0]

    num_of_features = trainset.shape[1]-1

    trainlabels = trainset[trainset.columns[num_of_features]]

    weights = np.zeros(num_of_features)

    training_examples = trainset[trainset.columns[:num_of_features]]

    bias = 0.0

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
            # computing dot product between weigts and training sample
            if label*(np.dot(weights,training_examples.ix[i])+bias) <=0:
                incorrect+=1
                #updating weights
                weights = weights + label*training_examples.ix[i]
                bias = bias + label
        print("Iteration:{} Incorrect:{}".format(iterations ,incorrect))
        if incorrect == 0:
            break

    print "\nBias:", bias
    print "Weights:",np.array(weights)
    print "Normalized Weights:",np.array(weights/(-bias))





