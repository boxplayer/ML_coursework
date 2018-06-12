import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import linalg

from cross_validation import cross_validation
from error_score import *
from import_data import import_data

from error_score import print_errors_3d_mtx, error_score, print_error_score

from cross_validation import create_cv_folds, train_and_test_partition
from results_plots import display_error_graphs

def main(wine_data):
    print("Polynomial Model")
    poly_model_reg(wine_data)

def poly_model_reg(wine_data):
    """
    Loops through a range of different regression coefficents to find the optimum
    then plots the error and comparison graphs
    """

    # Collects wine data and spilts accoriding to features (inputs) and labels (targets)
    # inputmtx = wine_data[:, 0:11]
    # print(wine_features[0,1,2,4,6,7,9,10])
    inputmtx = wine_data[:, (0,1,2,4,6,7,9,10)]
    targets = wine_data[:, 11]

    # Reserve data for model validation
    inputmtx, targets, final_inputs, final_targets = cross_validation(inputmtx, targets, 0.1)

    # Create Folds
    num_folds = 10
    N = len(targets)
    folds = create_cv_folds(N, num_folds)

    # Set variables and then train and test the model for each fold
    reg_coeffs = np.linspace(0, 1, 51)
    degrees = np.linspace(1, 5, 5)
    errors, weights = cv_evaluation_poly_model(inputmtx, targets, folds, degrees, reg_coeffs)

    # Collocate errors for each fold to and find the mean errors.
    errormean = np.zeros((len(reg_coeffs), len(degrees), 5))
    for fold in errors:
        matrix = errors[fold]
        errormean = matrix + errormean
    errormean = errormean / num_folds
    errormean = errormean[0, :, :, :]

    # Print Errors
    min_reg, min_deg = print_errors_3d_mtx(errormean, reg_coeffs, degrees)
    error_deg = errormean[0, :, :]
    display_error_graphs(error_deg, degrees, "Degree Factor", "Change in Polynomial Degrees", "deg")
    error_reg = errormean[:, 2, :]
    display_error_graphs(error_reg, reg_coeffs, "Regression Coefficient ($\lambda$)",
                         "Change in Regression Coefficient", "reg")

    # Create aggreate final model across all the folds:
    min_reg_index = reg_coeffs.tolist().index(min_reg)
    min_deg_index = degrees.tolist().index(min_deg)
    weightsksize = int(max(degrees) * inputmtx.shape[1])
    weightsmean = findbestweights(weights, min_reg_index, min_deg_index, weightsksize)

    test_optimised_model(min_deg, weightsmean, final_inputs, final_targets)


def cv_evaluation_poly_model(
        inputs, targets, folds, degrees, reg_coeffs):
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
    train_errors - the training errors for the approximation
    test_errors - the test errors for the approximation
    """

    # Creates dictionaries to store error and weight information for each fold
    errors = {}
    weights = {}
    for f, fold in enumerate(folds):
        errors["fold{0}".format(f)] = []
        weights["fold{0}".format(f)] = []

    for f, fold in enumerate(folds):
        print("Fold: ", f)
        # f is the fold id, fold is the train-test split
        train_part, test_part = fold
        # break the data into train and test sets
        train_inputs, train_targets, test_inputs, test_targets = \
            train_and_test_partition(inputs, targets, train_part, test_part)
        # trains and evaluate the error on both sets
        errormatrix, weightsmtx = polynomial(train_inputs, train_targets, test_inputs, test_targets, degrees,
                                             reg_coeffs)
        errors["fold{0}".format(f)].append(errormatrix)
        weights["fold{0}".format(f)] = weightsmtx
    return errors, weights


def polynomial(train_inputmtx, train_targets, test_inputmtx, test_targets, degrees, reg_coeffs):
    """
    Loops through a range of degrees and regression coefficients to find the optimum polynomial
    model
    :param train_inputmtx: training features
    :param train_targets: training labels
    :param test_inputmtx: testing features
    :param test_targets: testing labels
    :param degrees: array of degree factors to iterate through
    :param reg_coeffs: array of regression coeefiecnts to iterate through
    :return: Error and Weights matrices containing results of each iteration
    """
    weightsksize = int(max(degrees) * train_inputmtx.shape[1])
    errormtx = np.zeros((len(reg_coeffs), len(degrees), 5))
    weightsmtx = np.zeros((len(reg_coeffs), len(degrees), weightsksize))
    # Loop through Regression Coefficients
    for i in range(int(len(reg_coeffs))):
        # Loop through Degrees
        for j in range(int(len(degrees))):
            outputmtx = expand_to_2Dmonomials(train_inputmtx, int(degrees[j]))
            weights = regularised_ml_weights(outputmtx, train_targets, reg_coeffs[i])
            prediction_func = construct_3dpoly(int(degrees[j]), weights)
            prediction_values = prediction_func(test_inputmtx)
            # Evalute the prediction
            errorarr = error_score(test_targets, prediction_values)
            # Create 3D matrix of weights
            for k in range(len(weights)):
                weightsmtx[i, j, k] = weights[k]
            # Create 3D matrix of errors
            for k in range(0, 5):
                errormtx[i, j, k] = errorarr[k]

    return errormtx, weightsmtx


def regularised_ml_weights(inputmtx, targets, reg_coeff):
    """
    This method returns the regularised weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets), 1))
    I = np.identity(Phi.shape[1])
    weights = linalg.pinv(I * reg_coeff + Phi.transpose() * Phi) * Phi.transpose() * targets
    return np.array(weights).flatten()


def construct_3dpoly(degree, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """

    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        expanded_xs = np.matrix(expand_to_2Dmonomials(xs, degree))
        ys = expanded_xs * np.matrix(weights).reshape((len(weights), 1))
        return np.array(ys).flatten()

    # we return the function reference (handle) itself. This can be used like
    # any other function
    return prediction_function


def expand_to_2Dmonomials(inputs, degree):
    """
    Create a design matrix from a 2d array of input values, where columns
    of the output are powers of the inputs from 0 to degree (inclusive)

    So if input is: inputs=np.array([x1, x2, x3])  and degree = 4 then
    output will be design matrix:
        np.array( [[  1.    x1**1   x1**2   x1**3   x1**4   ]
                   [  1.    x2**1   x2**2   x2**3   x2**4   ]
                   [  1.    x3**1   x3**2   x3**3   x3**4   ]])
    """
    expanded_inputs = []
    for col in inputs.T:
        for i in range(degree):
            expanded_inputs.append(col ** i)
    return np.array(expanded_inputs).transpose()


def findbestweights(weights, min_reg_index, min_deg_index, weightsksize):
    """
    Finds the best weigths from each of the folds and aggregate this data together
    to create a final model
    :param weights: dictionary of weights values
    :param min_reg_index: minimum regression coeffcient value
    :param min_deg_index: minimum degree coefficient value
    :param weightsksize: size of the weights matrix
    :return: array of weights of final model
    """
    bestweights = np.zeros(weightsksize)
    for fold in weights:
        array = weights[fold]
        value = array[int(min_reg_index)][int(min_deg_index)][:]
        bestweights = np.vstack([bestweights, value])

    bestweights = np.delete(bestweights, 0, axis=0)
    weightsmean = np.trim_zeros(np.sum(bestweights, axis=0) / bestweights.shape[0])
    return weightsmean


def test_optimised_model(deg, weights, test_inputs, test_targets):
    """ Finds the error values for a trained model using already found
    weights and degrees"""
    prediction_func = construct_3dpoly(int(deg), weights)
    prediction_values = prediction_func(test_inputs)
    errorarr = error_score(test_targets, prediction_values)
    print_error_score(errorarr)


if __name__ == '__main__':
    main()
