from __future__ import division
import numpy as np
from scipy.spatial.distance import euclidean


def euclideanDist(datapoint,centroids):
    dist = []
    for centroid in centroids:
        dist.append(euclidean(datapoint,centroid))

    return dist

def recomputeCentroids(clusterGroup,centroids):

    new_centroids = []

    for x in xrange(clusterGroup.shape[0]):
        # print "indexes for group",x
        positions = np.where(clusterGroup[x,:] == 1)[0]
        print "Cluster group:",centroids[x]
        new_centroids.append(calculateCentroids(positions))

    return new_centroids

def calculateCentroids(positions):

    new_x = 0
    new_y = 0
    for position in positions:
        print datapoints[position]
        new_x+=datapoints[position][0]
        new_y+=datapoints[position][1]
    return (new_x/len(positions),new_y/len(positions))

def computeNewClusters(datapoints,centroids):

    distmatrix = []

    for point in datapoints:
        distmatrix.append(euclideanDist(point,centroids))

    distmatrix = np.array(distmatrix).transpose()

    clustergrp = np.zeros((len(centroids),len(datapoints)))

    # print clustergrp.shape[0]

    for i in xrange(distmatrix.shape[1]):
        minelement = min(distmatrix[:,i])
        index = np.where(distmatrix[:,i] == minelement)[0][0]
        # print datapoints[i],minelement,index
        clustergrp[:,i][index] = 1

    return clustergrp



if __name__ == '__main__':

    centroids = [(25,125), (44,105), (29,97), (35,63), (55,63), (42,57), (23,40), (64,37), (33,22),(55,20)]

    datapoints =  [(25,125), (44,105), (29,97), (35,63), (55,63), (42,57), (23,40), (64,37), (33,22),(55,20),(28,145),
                  (50,130),(65,140),(55,118),(38,115),(50,90),(63,88),(43,83),(50,60),(50,30)]


    prev = np.zeros((len(centroids),len(datapoints)))


    i=0

    print "Initial centroids:"
    print centroids


    while True:

        clusterGroup = computeNewClusters(datapoints,centroids)

        if np.array_equal(prev,clusterGroup):
            break

        centroids = recomputeCentroids(clusterGroup,centroids)

        prev = clusterGroup

        i+=1
        print "After Iteration:",i

        print centroids




