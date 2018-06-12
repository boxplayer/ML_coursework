import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import polyfit, polyval, linalg
from error_score import *
import scipy.stats as stats

from import_data import import_data
from cross_validation import cross_validation


def main():
    
    wine_data, wine_features = import_data('winequality-red.csv')
    
    #matrix of features
    inputmtx = wine_data[:, 0:11]

    #array of targets
    targets = wine_data[:, 11]
    N = len(targets)
    
    # set the k's and number of cross-v folds
    Ks = 160
    num_folds = 10
    
    # create cv folds
    folds = create_cv_folds(N, num_folds)
    
    # return an error of k x M rmse errors
    errormtxs = cv_evaluation(inputmtx, targets, folds, Ks)
    
    errormean = np.zeros((Ks, 5))
    for fold in errormtxs:
        matrix = errormtxs[fold]
        errormean = matrix + errormean

    errormean = errormean/num_folds

    threemtx = errormean[0, :, :]

    ks = np.linspace(1, Ks, Ks)
    display_error_graphs(threemtx, ks)

    print("AVERAGE ERRORS")
    print_errors_2d_mtx(threemtx, ks)
    
def knn_regression(train_inputs, train_targets, test_inputs, test_targets, Ks, fold_no):
    
    print('Train inputs: ', train_inputs.shape)
    print('Test inputs', test_inputs.shape)
    

    
    #create a matrix to store distances
    distances = np.zeros((len(train_inputs), len(test_inputs)))
    print("distances shape: ",distances.shape)
    
    #create a matrix of targets
    targets_list = np.zeros((len(train_inputs), len(test_inputs)))
    print("targets_list shape: ",targets_list.shape)
    
    
#    train_targets = np.array(training_set[:, 11])
    train_targets = np.array(train_targets)
    train_targets.tolist()
    ks = np.linspace(1, Ks, Ks)
    errormtx = np.zeros((len(ks), 5))
    
    train_inputs_len = len(train_inputs)
    test_inputs_len = len(test_inputs)
    
    
    
    # for each k
    for k in ks:
        
        k = int(k)
        kNN_targets = []
        
        # for each training row
        for test_row in range(0,test_inputs_len):
            # for each testing row compared to training row
#            if k == 1:
            for train_row in range(0, train_inputs_len):
                eud = euclidian_distance(test_inputs[test_row], train_inputs[train_row])
                distances[train_row][test_row] = eud
                targets_list[train_row][test_row] = train_targets[train_row]
                print("r")
            sort = [x for _, x in sorted(zip(distances[:,test_row], targets_list[:,test_row]))]
    #            print(k)
            kNN = sort[0:k]
            weighted_target = sum(kNN)/k        
            kNN_targets.append(weighted_target)
        errorarr = error_score(test_targets, kNN_targets)
        for m in range(0, 5):
            errormtx[k-1, m] = errorarr[m]

#    # for each k
#    for k in ks:
#        
#        k = int(k)
#        kNN_targets = []
#        
#        # for each training row
#        for train_row in range(0,train_inputs_len):
#            # for each testing row compared to training row
#            if k == 1:
#                for test_row in range(0, test_inputs_len):
#                    eud = euclidian_distance(train_inputs[train_row], test_inputs[test_row])
#                    distances[test_row][train_row] = eud
#                    targets_list[test_row][train_row] = test_targets[test_row]
##                if(len(distances[:,train_row]) != 1):
##                    print("len mismatch")
##                    print(distances[:,train_row].shape)
##                    print(targets_list[train_row].shape)
#                sort = [x for _, x in sorted(zip(distances[:,train_row], targets_list[train_row]))]
##            print(k)
#            kNN = sort[0:k]
#            weighted_target = sum(kNN)/k        
#            kNN_targets.append(weighted_target)
#        errorarr = error_score(train_targets, kNN_targets)
#        for m in range(0, 5):
#            errormtx[k-1, m] = errorarr[m]
        
        
#    # print(ks)
#    for k in ks:
#        k = int(k)
#        # print(k)
#        kNN_targets = []
#        for j in range(0, train_inputs.shape[0]):
#            if k == 1:
#                for i in range(0, test_inputs.shape[0]):
#                    eud = euclidian_distance(test_inputs[i, :], train_inputs[i, :])
#                    euclidian_distances[i][j] = eud
#                    targets_list[i][j] = target_set[i, 11]
#                sort = [x for _, x in sorted(zip(euclidian_distances[:, j], targets_list[:, j]))]
#            kNN = sort[0:k]
#            weighted_target = sum(kNN)/k
#            kNN_targets.append(weighted_target)
#        errorarr = error_score(train_targets, kNN_targets)
#        for m in range(0, 5):
#            errormtx[k-1, m] = errorarr[m]


    # print errors
    
    print_errors_2d_mtx(errormtx, ks)
    display_error_graphs(errormtx, ks)

    # print graphs
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ks, errormtx[:, 0], 'r')
    ax.set_xlabel("k")
    ax.set_ylabel("RMSE")
    plt.title("k vs RMSE")
    plt.show()
    fig.savefig("kNN_graphs/knnRMSE_fold" + str(fold_no) + ".pdf", fmt="pdf")
    
    
    # return rmse values / error values
#    return errormtx[:, 0]
    return errormtx

def cv_evaluation(
        inputs, targets, folds, Ks, reg_param=None):
    """
    Will split inputs and targets into train and test parts, then fit a linear
    model to the training part, and test on the both parts.

    Inputs can be a data matrix (or design matrix), targets should
    be real valued.

    parameters
    ----------
    inputs - the input design matrix (any feature mapping should already be
        applied)
    targets - the targets as a vector
    num_folds - the number of folds
    reg_param (optional) - the regularisation strength. If provided, then
        regularised least squares fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_inputs
    train_targets
    test_inputs
    test_targets
    
    
    """
    
    errors = {}
    for f, fold in enumerate(folds):
        errors["fold{0}".format(f)] = []
    
    # get the number of datapoints
    N = inputs.shape[0]
    # get the number of folds
    num_folds = len(folds)
    train_errors = np.empty(num_folds)
    test_errors = np.empty(num_folds)
    
    kM_errormtx = np.zeros((Ks,num_folds))
    
    for f,fold in enumerate(folds):
        # f is the fold id, fold is the train-test split
        train_part, test_part = fold
        # break the data into train and test sets
        train_inputs, train_targets, test_inputs, test_targets = \
            train_and_test_partition(inputs, targets, train_part, test_part)

        print(train_inputs.shape)
        print(test_inputs.shape)
        
        # get rmse values
#        errors = knn_regression(training_set, target_set, Ks, f)
        threematrix = knn_regression(
                train_inputs, train_targets, test_inputs, test_targets, Ks, f)
        
        
        #put them in the error matrix
        errors["fold{0}".format(f)].append(threematrix)
#        kM_errormtx[:,f] = errors

        

#    return kM_errormtx
    return errors

def euclidian_distance(vec1, vec2):
    joined_vectors = zip(vec1, vec2)
    distance = 0
    for element in joined_vectors:
        distance += (element[1] - element[0]) ** 2
    
    return math.sqrt(distance)

def manhattan_distance(vec1, vec2):
    joined_vectors = zip(vec1, vec2)
    distance = 0
    for element in joined_vectors:
        distance += abs(element[1] - element[0])
    return distance


def create_cv_folds(N, num_folds):
    """
    Defines the cross-validation splits for N data-points into num_folds folds.
    Returns a list of folds, where each fold is a train-test split of the data.
    Achieves this by partitioning the data into num_folds (almost) equal
    subsets, where in the ith fold, the ith subset will be assigned to testing,
    with the remaining subsets assigned to training.

    parameters
    ----------
    N - the number of datapoints
    num_folds - the number of folds

    returns
    -------
    folds - a sequence of num_folds folds, each fold is a train and test array
        indicating (with a boolean array) whether a datapoint belongs to the
        training or testing part of the fold.
        Each fold is a (train_part, test_part) pair where:

        train_part - a boolean vector of length N, where if ith element is
            True if the ith data-point belongs to the training set, and False if
            otherwise.
        test_part - a boolean vector of length N, where if ith element is
            True if the ith data-point belongs to the testing set, and False if
            otherwise.
    """
    # if the number of datapoints is not divisible by folds then some parts
    # will be larger than others (by 1 data-point). min_part is the smallest
    # size of a part (uses integer division operator //)
    min_part = N//num_folds
    # rem is the number of parts that will be 1 larger
    rem = N % num_folds
    # create an empty array which will specify which part a datapoint belongs to 
    parts = np.empty(N, dtype=int)
    start = 0
    for part_id in range(num_folds):
        # calculate size of the part
        n_part = min_part
        if part_id < rem:
            n_part += 1
        # now assign the part id to a block of the parts array
        parts[start:start+n_part] = part_id*np.ones(n_part)
        start += n_part
    # now randomly reorder the parts array (so that each datapoint is assigned
    # a random part.
    np.random.shuffle(parts)
    # we now want to turn the parts array, into a sequence of train-test folds
    folds = []
    for f in range(num_folds):
        train = (parts != f)
        test = (parts == f)
        folds.append((train,test))
    return folds


def train_and_test_partition(inputs, targets, train_part, test_part):
    """
    Splits a data matrix (or design matrix) and associated targets into train
    and test parts.

    parameters
    ----------
    inputs - a 2d numpy array whose rows are the datapoints, or can be a design
        matric, where rows are the feature vectors for data points.
    targets - a 1d numpy array whose elements are the targets.
    train_part - A list (or 1d array) of N booleans, where N is the number of
        data points. If the ith element is true then the ith data point will be
        added to the training data.
    test_part - (like train_part) but specifying the test points.

    returns
    -------     
    train_inputs - the training input matrix
    train_targets - the training targets
    test_inputs - the test input matrix
    test_targets - the test targtets
    """
    # get the indices of the train and test portion
    if len(inputs.shape) == 1:
        # if inputs is a sequence of scalars we should reshape into a matrix
        inputs = inputs.reshape((inputs.size,1))
    train_inputs = inputs[train_part,:]
    test_inputs = inputs[test_part,:]
    train_targets = targets[train_part]
    test_targets = targets[test_part]
    return train_inputs, train_targets, test_inputs, test_targets

def construct_knn_approx(train_inputs, train_targets, k):  
    """
    For 1 dimensional training data, it produces a function: reals-> reals
    that outputs the mean training value in the k-Neighbourhood of any input.
    """
    # Create Euclidean distance.
    distance = lambda x,y: (x-y)**2
    train_inputs = np.resize(train_inputs, (1,train_inputs.size))
    def prediction_function(inputs):
        inputs = inputs.reshape((inputs.size,1))
        print(inputs.shape)
        distances = distance(train_inputs, inputs)
        predicts = np.empty(inputs.size)
        for i, neighbourhood in enumerate(np.argpartition(distances, k)[:,:k]):
            # the neighbourhood is the indices of the closest inputs to xs[i]
            # the prediction is the mean of the targets for this neighbourhood
            predicts[i] = np.mean(train_targets[neighbourhood])
        return predicts
    # We return a handle to the locally defined function
    return prediction_function

def conjoin_data_for_knn(train_inputmtx, train_targets, test_inputmtx, test_targets):
    train_targets = train_targets
    train_inputmtx = np.array(train_inputmtx).T
    training_set = np.vstack((train_inputmtx, train_targets)).T
    test_targets = test_targets
    test_inputmtx = np.array(test_inputmtx).T
    target_set = np.vstack((test_inputmtx, test_targets)).T
    
    return target_set, training_set

def display_error_graphs(threemtx, Ks):
    error_name = ["RMS", "Mean", "Median", "Mean Percentage", "Variance"]
    for type in range(0, 5):
        error_arr = threemtx[:, type]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(Ks, error_arr, 'b')
        ax.set_xlabel("K")
        ax.set_ylabel(error_name[type])
        plt.title(error_name[type] + " vs K")
        fig.savefig("kNN_graphs/" + error_name[type] + "vsDegree.pdf", fmt="pdf")
        plt.show()


if __name__ == '__main__':

  main()