import numpy as np
import pandas as pd



'''
Activation function
'''

def sigmoid(x):
    return 1.0/float(1.0+np.exp(-x))

'''
Differential of activation function
'''

def diffsigmoid(Ok):
    return Ok*(1.0 - Ok)

'''
Converting labels to 1 and 0
'''

def convertLabels(label):
    if label == 5:
        return 1
    else:
        return 0



class MLP:



    def __init__(self,input_nodes,hidden_nodes,output_nodes):

        '''

        :param input_nodes: Number of Nodes in Input layer
        :param hidden_nodes: Number of Nodes in Hidden Layer
        :param output_nodes: Number of Nodes in Output Layer
        :return: -
        '''


        self.input_layer = input_nodes+1  # +1 for bias
        self.hidden_layer = hidden_nodes
        self.output_layer = output_nodes

        '''
        Initialising input weights vector and output weights vector
        '''

        self.iweights = np.random.uniform(-0.2,0.2,(self.input_layer,self.hidden_layer))
        self.oweights = np.random.uniform(-1.0,1.0,(self.hidden_layer,self.output_layer))

        '''
        Initialising activation functions at each layer
        '''

        self.activation_in = np.ones(self.input_layer)
        self.activation_hidden = np.ones(self.hidden_layer)
        self.activation_out = np.ones(self.output_layer)

        self.momentum_in = np.zeros((self.input_layer,self.hidden_layer))
        self.momentum_out = np.zeros((self.hidden_layer,self.output_layer))


        self.grad_sigmoid = np.vectorize(diffsigmoid)
        self.sigmoidfn = np.vectorize(sigmoid)



    def updateNode(self,sample):

        '''
        Feed forwarding the Neural network

        :param sample: Training Sample
        :return: output activation value
        '''

        self.activation_in[:self.input_layer-1] = sample

        # hidden layer activation values
        self.activation_hidden = self.sigmoidfn(np.dot(self.activation_in,self.iweights))


        # output layer activation value
        self.activation_out = self.sigmoidfn(np.dot(self.activation_hidden,self.oweights))

        return self.activation_out

    def BackPropagate(self,label,N=0.5,M=0.1):

        '''

        :param label: True Label
        :param N: Learning rate
        :param M: Momentum factor
        :return: Error
        '''


        #starting with output unit
        output_delta = diffsigmoid(self.activation_out) * (label - self.activation_out)

        # Hidden unit errors
        hidden_errors = np.multiply(output_delta,self.oweights)
        hidden_deltas = np.multiply(hidden_errors,self.grad_sigmoid(self.activation_hidden).reshape(self.hidden_layer,1))


        # Updating output weights
        hchanges = np.multiply(N,np.multiply(output_delta,self.activation_hidden)).reshape(self.hidden_layer,1)
        h_momentum = np.multiply(M,self.momentum_out)

        self.oweights = np.add(self.oweights,hchanges)
        self.oweights = np.add(self.oweights,h_momentum)
        self.momentum_out = hchanges


        #updating input weights
        ichanges = np.multiply(N,np.multiply(hidden_deltas,self.activation_in).T)
        i_momentum = np.multiply(M,self.momentum_in)

        self.iweights = np.add(self.iweights,ichanges)
        self.iweights = np.add(self.iweights,i_momentum)
        self.momentum_in = ichanges



        # Calculating Error
        return 0.5*((label-self.activation_out)**2)


    def train(self,dataset,labels,iterations=100):

        '''

        :param dataset: Training Dataset
        :param labels: Training True labels
        :param iterations: Number of times to run the algorithm
        :return: -
        '''

        for l in range(1,iterations+1):

            error = 0.0
            for i in range(0,dataset.shape[0]):
                sample = dataset[i]
                label = labels[i]
                self.updateNode(sample)
                error += self.BackPropagate(label)

            # Printing Errors for every 10 Iterations
            if l%10 == 0:
                print("After {} Iterations, Error value: {:.5f}".format(l ,error[0]))



    def test(self, test_input,test_labels):
        '''

        :param test_input: Test dataset
        :param test_labels: Test data True labels
        :return: Accuracy of the model
        '''
        correct = 0
        errors = 0
        for i in range(0,test_input.shape[0]):
            if self.updateNode(test_input[i]) >= 0.5 and test_labels[i] == 1:
                    correct+=1
            elif self.updateNode(test_input[i]) < 0.5 and test_labels[i] == 0:
                    correct+=1
            else:
                errors+=1

        print("\n\t\tTotal Test Data: {}\n\t\tCorrect Predictions: {}\n\t\tWrong Predictions: {}\n"
              "\t\tAccuracy: {:.2f}%\n".format(test_input.shape[0],correct,errors,
                                               (correct/float(test_input.shape[0]))*100))

        print "*"*41





if __name__ == '__main__':


    '''
    Reading the Train and Test Data and processing them
    '''

    trainset = pd.read_csv("a2_datasets/digits/train.csv")
    testset = pd.read_csv("a2_datasets/digits/test.txt")

    # Converting the labels of the data
    trainlabels = np.array(trainset['label'].apply(convertLabels))
    testlabels = np.array(testset['label'].apply(convertLabels))

    # Normalizing the Data to have values between 0 and 1
    training_data = np.array(trainset.iloc[0:,1:]/255.0)
    test_data = np.array(testset.iloc[0:,1:]/255.0)

    # For different number of hidden layer nodes
    for hn in [10,20,30,50,100]:

        print "\n\t\t\tHidden Nodes: {}".format(hn)

        bp = MLP(training_data.shape[1],hn,1)

        bp.train(training_data,trainlabels)

        bp.test(test_data,testlabels)






