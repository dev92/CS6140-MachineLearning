import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score


'''
Converting labels to 1 and -1
'''

def convertLabels(label):
    if label == 5:
        return 1
    else:
        return -1



class SSMO:

    def __init__(self,num_of_examples,feature_len,C=0.01,tol=0.5,passes=3):
        self.C = C
        self.tol = tol
        self.max_passes = passes

        self.bias = 0.0
        self.E_i = 0.0
        self.E_j = 0.0
        self.old_alpha_i = 0
        self.old_alpha_j = 0
        self.L = 0.0
        self.H = 0.0
        self.eta  = 0.0
        self.alphas = np.zeros(num_of_examples)
        # self.ratio_alpha_changes = 0.01
        self.weights = np.zeros(feature_len)


    '''
    Applying Linear kernel K(x,x') = inner product
    '''

    def linearKernel(self,u,v):
        return np.dot(u,v)

    '''
    Computing F(x) = summation of alphas * labels * K(x,x')
    '''

    def computeFx(self,matrix,labels,alphas,b,index):
        result = np.multiply(alphas,labels)
        result = np.dot(result,self.linearKernel(matrix,matrix[index].T))
        return float(result + b)


    '''
    Computing Eta value
    '''


    def computeEta(self,dataSet,i,j):
        kij = self.linearKernel(dataSet[i],dataSet[j])
        kii = self.linearKernel(dataSet[i],dataSet[i])
        kjj = self.linearKernel(dataSet[j],dataSet[j])
        return (2.0 * kij) - kii - kjj


    '''
    Updating Alpha[j] value
    '''

    def computeAlphaJ(self,j,labels):
        self.alphas[j] -= float(labels[j] * (self.E_i - self.E_j)/self.eta)

        if self.alphas[j] > self.H:
            self.alphas[j] = self.H
        elif self.alphas[j] < self.L:
            self.alphas[j] = self.L
        else:
            self.alphas[j] = self.alphas[j]


    '''
    Updating Bias value
    '''

    def computeBias(self,labels,i,dataSet,j):

        bias1 = self.bias - self.E_i - (labels[i]*(self.alphas[i] - self.old_alpha_i) * \
                                   (self.linearKernel(dataSet[i],dataSet[i])) - \
                                   labels[j]*(self.alphas[j] - self.old_alpha_j) * (self.linearKernel(dataSet[i],dataSet[j])))

        bias2 = self.bias - self.E_j - (labels[i]*(self.alphas[i] - self.old_alpha_i) * \
                                   (self.linearKernel(dataSet[i],dataSet[j])) - \
                                   labels[j]*(self.alphas[j] - self.old_alpha_j) * (self.linearKernel(dataSet[j],dataSet[j])))

        if 0 < self.alphas[i] < self.C:
            return bias1
        elif 0 < self.alphas[j] < self.C:
            return bias2
        else:
            return  (bias1 + bias2)/2.0


    '''
    Calculating Weights vector based on updated alphas
    '''

    def calculateWeightsVector(self,dataMatrix,label):
        self.weights = np.dot(np.multiply(self.alphas,label),dataMatrix)



    '''
    Training Algorithm starts here, iterates until convergence is reached max_passess time
    '''


    def train(self,dataSet,labels):


        passes = 0
        sample_size = dataSet.shape[0]

        while(passes < self.max_passes):

            num_changed_alphas = 0

            for i in range(0,sample_size):

                self.E_i  = self.computeFx(dataSet,labels,self.alphas,self.bias,i) - labels[i]

                if((labels[i] * self.E_i < -self.tol and self.alphas[i] < self.C) or
                       (labels[i] * self.E_i > self.tol and self.alphas[i] > 0)):

                    j = np.random.randint(0,sample_size)
                    while j==i:
                        j = np.random.randint(0,sample_size)

                    self.E_j = self.computeFx(dataSet,labels,self.alphas,self.bias,j) - labels[j]

                    self.old_alpha_i = self.alphas[i]
                    self.old_alpha_j = self.alphas[j]

                    if labels[i] != labels[j]:
                        self.L = max(0, self.alphas[j] - self.alphas[i])
                        self.H = min(self.C, self.C + (self.alphas[j] - self.alphas[i]))
                    else:
                        self.L = max(0, (self.alphas[j] + self.alphas[i]) - self.C)
                        self.H = min(self.C, self.alphas[j] + self.alphas[i])

                    if self.L == self.H:
                        continue

                    self.eta = self.computeEta(dataSet,i,j)

                    if self.eta >= 0:
                        continue

                    self.computeAlphaJ(j,labels)

                    if abs(self.alphas[j] - self.old_alpha_j) < 1e-5:
                        continue

                    self.alphas[i] = self.old_alpha_i + (labels[i]*labels[j]*(self.old_alpha_j - self.alphas[j]))

                    #updating bias
                    self.bias = self.computeBias(labels,i,dataSet,j)

                    num_changed_alphas += 1



            print("Pass:{}, alphas changed: {}".format(passes, num_changed_alphas))

            # if(num_changed_alphas <= self.ratio_alpha_changes*sample_size):
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        self.calculateWeightsVector(dataSet,labels)



    '''
    Computing accuracy of the trained model, using Test Data
    '''


    def test(self,testData,test_labels):

        predicted_labels = np.dot(self.weights,testData.T)
        correct = 0
        errors = 0

        for index in range(testData.shape[0]):
            if predicted_labels[index] + self.bias >= 0 and test_labels[index] == 1:
                correct+=1
            elif predicted_labels[index] + self.bias < 0 and test_labels[index] == -1:
                correct+=1
            else:
                errors+=1


        print "\nLinear Kernel SVM using Simplified SMO (Own Implementation)"

        print("\n\t\tTotal Test Data: {}\n\t\tCorrect Predictions: {}\n\t\tWrong Predictions: {}\n"
              "\t\tAccuracy: {:.2f}%\n".format(testData.shape[0],correct,errors, (correct/float(testData.shape[0]))*100))

        print "*"*40


'''
Using Sci-kit library to compare the results
'''

def librarySVM(trainData,testData,trainLabels,testLabels):

    svm_linear = svm.SVC(kernel='linear')
    svm_rbf = svm.SVC(kernel='rbf')
    svm_poly = svm.SVC(kernel='poly')

    print "\n\t\tSVM using Sci-Kit Library\n"

    svm_linear.fit(trainData,trainLabels)
    linear_predicted = svm_linear.predict(testData)
    print("\tLinear Kernel SVM - Accuracy: {:.2f}%".format(accuracy_score(testLabels,linear_predicted) * 100))

    svm_rbf.fit(trainData,trainLabels)
    rbf_predicted = svm_rbf.predict(testData)
    print("\tGuassian Kernel SVM - Accuracy: {:.2f}%".format(accuracy_score(testLabels,rbf_predicted) * 100))

    svm_poly.fit(np.multiply(255.0,trainData),trainLabels)
    poly_predicted = svm_poly.predict(np.multiply(255.0,testData))
    print("\tPolynomial Kernel SVM - Accuracy: {:.2f}%".format(accuracy_score(testLabels,poly_predicted) * 100))



if __name__ == '__main__':


    '''
    Reading the Train and Test Data and processing them
    '''


    trainset = pd.read_csv("a2_datasets/digits/train.csv")
    testset = pd.read_csv("a2_datasets/digits/test.txt")

    num_of_examples = 2000   # Number of Train samples to use

    # Converting the labels of the data
    trainlabels = np.array(trainset['label'].apply(convertLabels))
    testlabels = np.array(testset['label'].apply(convertLabels))

    # Normalizing the Data to have values between 0 and 1
    training_data = np.array(trainset.iloc[0:,1:]/255.0)
    test_data = np.array(testset.iloc[0:,1:]/255.0)



    Svm = SSMO(num_of_examples,training_data.shape[1],0.01,0.5,3)

    Svm.train(training_data[range(0,num_of_examples),:],trainlabels[:num_of_examples])
    Svm.test(test_data,testlabels)

    librarySVM(training_data,test_data,trainlabels,testlabels)

