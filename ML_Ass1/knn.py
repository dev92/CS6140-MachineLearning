__author__ = 'Dev'
import pandas as pd
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



df = pd.read_csv("knn_2D/train.data")
test_set = pd.read_csv("knn_2D/test.data")

train_acc = []
test_acc = []

for i in range(1,13):

    clf = neighbors.KNeighborsClassifier(i)

    clf.fit(df[['x1','x2']].values,df['label'].values)

    predicted = clf.predict(test_set[['x1','x2']].values)
    train_predicted = clf.predict(df[['x1','x2']].values)

    error = 0
    correct = 0



    for x,y in zip(predicted,test_set['label'].values):
        if x!=y:
            error+=1
        else:
            correct+=1

    test_acc.append((correct/float(correct+error))*100)
    # print "K=",i , "Wrong:",error,"Correct:",correct,(correct/float(correct+error))*100

    error = 0
    correct = 0


    for x,y in zip(train_predicted,df['label'].values):
        if x!=y:
            error+=1
        else:
            correct+=1

    train_acc.append((correct/float(correct+error))*100)


n_groups = 12
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}
plt.xlabel('K value')
plt.ylabel('Accuracy (%)')
plt.title('Accuracies on the training and test sets')

rects1 = plt.bar(index, train_acc, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='Train Accuracy')

rects2 = plt.bar(index + bar_width, test_acc, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='Test Accuracy')
plt.xticks(index + bar_width, range(1,13))
plt.legend()
plt.tight_layout()
plt.show()







