import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import linalg
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

from cross_validation import cross_validation
from error_score import *
from import_data import import_data


def main():
    poly_model_reg()


def poly_model_reg():
    """
    Loops through a range of different regression coefficents to find the optimum
    then plots the error and comparison graphs
    """
    wine_data, wine_features = import_data('winequality-red.csv')
    inputmtx = wine_data[:, 0:11]
    # inputmtx = wine_data[:, [1,9,10]] # For the improved Regression Answer
    targets = wine_data[:, 11]
    train_inputmtx, train_targets, test_inputmtx, test_targets = \
        cross_validation(inputmtx, targets, 0.25)
    reg_coeffs = np.linspace(0, 1, 51)
    degrees = np.linspace(1, 15, 15)
    errormtx = np.zeros((len(reg_coeffs), len(degrees)))
    threemtx = np.zeros((len(reg_coeffs), len(degrees), 5))

    for i in range(int(len(errormtx))):
        for j in range(int(len(errormtx[i]))):
            outputmtx = expand_to_2Dmonomials(train_inputmtx, int(degrees[j]))
            weights = regularised_ml_weights(outputmtx, train_targets, reg_coeffs[i])
            prediction_func = construct_3dpoly(int(degrees[j]), weights)
            prediction_values = prediction_func(test_inputmtx)
            errorarr = error_score(test_targets, prediction_values)
            for k in range(0, 5):
                threemtx[i, j, k] = errorarr[k]
    
    display_error_graphs(threemtx, degrees)
    display_3d_error_graphs(threemtx, reg_coeffs, degrees)
    
    
    print_errors_3d_mtx(threemtx, reg_coeffs, degrees)


def ml_weights(inputmtx, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    PhiT = Phi.transpose()
    targets = np.matrix(targets).reshape((len(targets), 1))
    weights = linalg.pinv(PhiT * Phi) * PhiT * targets
    return np.array(weights).flatten()


def regularised_ml_weights(inputmtx, targets, reg_coeff):
    """
    This method returns the regularised weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    I = np.identity(Phi.shape[1])
    weights = linalg.pinv(I*reg_coeff + Phi.transpose()*Phi)*Phi.transpose()*targets
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
    Create a design matrix from a 1d array of input values, where columns
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
            expanded_inputs.append(col**i)
    return np.array(expanded_inputs).transpose()


def display_error_graphs(threemtx, degrees):
    error_name = ["RMS", "Mean", "Median", "Mean Percentage", "Variance"]
    for type in range(0, 5):
        error_arr = threemtx[:, :, type]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(degrees, error_arr[0,:], 'b')
        ax.set_xlabel("Degree")
        ax.set_ylabel(error_name[type])
        plt.title(error_name[type] + " vs Degree")
        fig.savefig("poly_graphs/" + error_name[type] + "vsDegree.pdf", fmt="pdf")
        plt.show()
    
def display_3d_error_graphs(threemtx, reg_coeffs, degrees):
    error_name = ["RMS", "Mean", "Median", "Mean Percentage", "Variance"]
    for type in range(0, 5):
        error_arr = threemtx[:, :, type]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Degree")
        ax.set_ylabel(error_name[type])
        ax.set_ylim(0.3,0.8)
        ax.set_xlim(0, 6)
        plt.title(error_name[type] + " vs Degree for each reg_coeff")
        for coef in range(0,50):
            ax.plot(degrees, error_arr[coef,:], linewidth=0.1)
        fig.savefig("poly_graphs/coeff_" + error_name[type] + "vsDegree.pdf", fmt="pdf")
        plt.show()
        
    
if __name__ == '__main__':

  main()
    