import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import polyfit, polyval, linalg
from Models.error_score import *
import scipy.stats as stats

from import_data import import_data
from cross_validation import cross_validation

from Models.error_score import rms_error


def main():
    wine_data, wine_features = import_data('winequality-red.csv')
    inputmtx = wine_data[:, 0:11]
    targets = wine_data[:, 11]
    train_inputmtx, train_targets, test_inputmtx, test_targets = cross_validation(inputmtx, targets, 0.5)
    train_targets = train_targets
    train_inputmtx = np.array(train_inputmtx).T
    training_set = np.vstack((train_inputmtx, train_targets)).T
    test_targets = test_targets
    test_inputmtx = np.array(test_inputmtx).T
    target_set = np.vstack((test_inputmtx, test_targets)).T
    print('Train Data Size: ', training_set.shape)
    print('Test', target_set.shape)
    euclidian_distances = np.zeros((len(target_set), len(training_set)))
    targets_list = np.zeros((len(target_set), len(training_set)))
    train_targets = np.array(training_set[:, 11])
    train_targets.tolist()
    ks = np.linspace(1, 800, 800)
    errormtx = np.zeros((len(ks), 5))
    # print(ks)
    for k in ks:
        k = int(k)
        # print(k)
        kNN_targets = []
        for j in range(0, training_set.shape[0]):
            if k == 1:
                for i in range(0, target_set.shape[0]):
                    eud = euclidian_distance(training_set[i, :], target_set[i, :])
                    euclidian_distances[i][j] = eud
                    targets_list[i][j] = target_set[i, 11]
                sort = [x for _, x in sorted(zip(euclidian_distances[:, j], targets_list[:, j]))]
            kNN = sort[0:k]
            weighted_target = sum(kNN)/k
            kNN_targets.append(weighted_target)
        errorarr = error_score(train_targets, kNN_targets)
        for m in range(0, 5):
            errormtx[k-1, m] = errorarr[m]

    print_errors_2d_mtx(errormtx, ks)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ks, errormtx[:, 0], 'r')
    ax.set_xlabel("k")
    ax.set_ylabel("rmse")
    plt.show()


def euclidian_distance(vec1, vec2):
    joined_vectors = zip(vec1, vec2)
    distance = 0
    for element in joined_vectors:
        distance += (element[1] - element[0]) ** 2
    
    return math.sqrt(distance)


if __name__ == '__main__':

  main()