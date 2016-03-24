import numpy as np


def splitAt(split):

    print
    print "Split at {}:".format(split)
    print "Cluster A vertices:",np.add(np.where(eigen_vectors[:,index] > split)[0],1)
    print "Cluster B vertices:",np.add(np.where(eigen_vectors[:,index] < split)[0],1)

    if split != 0:
        print "vertex ",np.where(eigen_vectors[:,index] == split)[0][0]+1," is randomly assigned to any one cluster"



if __name__ == '__main__':

    A = np.array([[0,4,3,0,0],
                  [4,0,2,0,0],
                  [3,2,0,1,2],
                  [0,0,1,0,0],
                  [0,0,2,0,0]])

    D = np.array([[7,0,0,0,0],
                  [0,6,0,0,0],
                  [0,0,8,0,0],
                  [0,0,0,1,0],
                  [0,0,0,0,2]])

    L = D - A

    print "Laplacian Matrix:"
    print L

    print

    result = np.linalg.eig(L)

    eigen_values,eigen_vectors = np.round(result[0],3),np.round(result[1],3)

    print "Eigen values:"
    print eigen_values

    print

    print "Eigen vectors associated:"
    print eigen_vectors

    print

    index = np.where(eigen_values == sorted(eigen_values)[1])[0][0]

    print "Second lowest Eigen value:",eigen_values[index]
    print
    print "Eigen vector associated with it:"
    print eigen_vectors[:,index]


    splitAt(0)

    print
    median = np.median(eigen_vectors[:,index])
    print "Median of the Eigen vector is:",median

    splitAt(median)

