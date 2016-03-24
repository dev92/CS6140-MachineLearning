import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def getScatterPlot():

    plt.figure()
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    for c, i, target_name in zip(colors.cnames, xrange(10), xrange(10)):
        plt.scatter(transformed_trainData[np.array(trainLabels == i),0][:100],
                    transformed_trainData[np.array(trainLabels == i),1][:100], c=c, label="digit "+target_name.__str__())
    plt.legend()
    plt.title('PCA of Digits dataset with 2 components')
    plt.savefig("scatterplot.png")


if __name__ == '__main__':


    components = [2,5,10,20,50,100,200]

    trainData = pd.read_csv('a3_datasets/digits/train.csv')
    testData = pd.read_csv('a3_datasets/digits/test.csv')

    samples = int(len(trainData)*0.2)

    trainFeatures = np.array(trainData.iloc[:samples,1:])
    trainLabels = trainData['label'][:samples]
    testFeatures = np.array(testData.iloc[:,1:])
    testLabels = np.array(testData['label'])

    accuracy_scores = []

    for n in components:


        pca = PCA(n_components = n, whiten = True)
        pca.fit(trainFeatures)
        transformed_trainData = pca.transform(trainFeatures)
        transformed_testData = pca.transform(testFeatures)

        if n == 2:
            getScatterPlot()


        clf = SVC(kernel='linear')
        clf.fit(transformed_trainData, trainLabels)
        pred = clf.predict(transformed_testData)

        score = accuracy_score(testLabels, pred)*100
        print("Accuracy of linear Kernel with {} components is : {:.2f}%  ".format(n,score))
        accuracy_scores.append(score)




    plt.figure()
    plt.xlabel('N components')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Vs Components on Digits dataset')
    plt.plot(components,accuracy_scores,color = 'blue' ,marker='x')
    plt.savefig("AccuracyPlot.png")

    print "Plot image files generated"
    # plt.show()




