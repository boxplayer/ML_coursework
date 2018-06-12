import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import linalg
import pandas as pd

from cross_validation import cross_validation
from import_data import import_data
from error_score import *


def main():
    poly_model_reg()


def poly_model_reg():
    """
    Loops through a range of different regression coefficents to find the opitimun
    then plots the error and comparision graphs
    """
    wine_data, wine_features = import_data('winequality-red.csv')
    inputmtx = wine_data[:, 0:10]
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
    """
    Bayes Addition
    """
    min_rmse, min_regcoef, min_polycoef = print_errors_3d_mtx(threemtx, reg_coeffs, degrees)
    outputmtx = expand_to_2Dmonomials(train_inputmtx, int(min_polycoef))
    M = outputmtx.shape[1]
    # define a prior mean and covaraince matrix
    # m0 = np.zeros(M)
    m0 = np.repeat(0, M)
    alphas = np.linspace(1, 100, 100)

    """
    Deduce Beta

    1. Via variance of raw targets 
    2. Via variance of residuals acquired by linear regression 
    """
    lin_weights = ml_weights(train_inputmtx, train_targets)
    prediction_values = linear_model_predict(test_inputmtx, lin_weights)
    resdiduals = test_targets - prediction_values
    resdidual_var = np.var(resdiduals)
    beta2 = (1. / resdidual_var) ** 2
    print(beta2)

    raw_var = np.var(targets)
    beta = (1. / raw_var) ** 2
    print(beta)
    """"""
    rsmes = []
    for x in range(1, int(len(alphas))):
        S0 = x * np.identity(M)
        mN, SN = calculate_weights_posterior(outputmtx, train_targets, beta, m0, S0)
        prediction_func = construct_3dpoly(int(min_polycoef), mN)
        prediction_values = prediction_func(test_inputmtx)
        rmse2, mae, medae, mape, variance = error_score(test_targets, prediction_values)
        rsmes.append(rmse2)
    index_min_rc = rsmes.index(min(rsmes))
    print("Best RSME for Bayesian: ", min(rsmes))
    print("Best Alpha for Bayesian Prior: ", index_min_rc)

    S0 = 98 * np.identity(M)
    mN, SN = calculate_weights_posterior(outputmtx, train_targets, beta, m0, S0)
    prediction_func = construct_3dpoly_bayesian(int(min_polycoef), mN, beta, SN)
    prediction_values, sigma2Ns = prediction_func(test_inputmtx)
    lower = prediction_values - np.sqrt(sigma2Ns)
    upper = prediction_values + np.sqrt(sigma2Ns)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Predicted Y")
    ax.set_ylabel("Upper and lower bounds of Y")
    ax.plot(prediction_values, lower, 'bx')
    ax.plot(prediction_values, upper, 'rx')
    plt.show()
    print(lower, upper)

    """"""


def predictive_dist(beta, designmtx, SN):
    N, K = designmtx.shape
    Phi = np.matrix(designmtx)
    # Phi = np.matrix(designmtx).reshape((K,1))
    SN = np.matrix(SN)
    sigma2Ns = np.ones(N) / beta
    for n in range(N):
        # now calculate and add in the data dependent term
        # NOTE: I couldn't work out a neat way of doing this without a for-loop
        # NOTE: but if anyone can please share the answer.
        phi_n = Phi[n, :].transpose()
        sigma2Ns[n] += phi_n.transpose() * SN * phi_n
    return np.array(sigma2Ns)


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
    targets = np.matrix(targets).reshape((len(targets), 1))
    I = np.identity(Phi.shape[1])
    weights = linalg.pinv(I * reg_coeff + Phi.transpose() * Phi) * Phi.transpose() * targets
    return np.array(weights).flatten()


def calculate_weights_posterior(designmtx, targets, beta, m0, S0):
    """
    Calculates the posterior distribution (multivariate gaussian) for weights
    in a linear model.

    parameters
    ----------
    designmtx - 2d (N x K) array of inputs (data-matrix or design-matrix) where
        N is the number of data-points and each row is that point's
        representation
    targets - 1d (N)-array of target values
    beta - the known noise precision
    m0 - prior mean (vector) 1d-array (or array-like) of length K
    S0 - the prior covariance matrix 2d-array

    returns
    -------
    mN - the posterior mean (vector) weight
    SN - the posterior covariance matrix


    weights = linalg.pinv(PhiT * Phi) * PhiT * targets
    """
    N, K = designmtx.shape
    Phi = np.matrix(designmtx)
    t = np.matrix(targets).reshape((N, 1))
    m0 = np.matrix(m0).reshape((K, 1))
    S0_inv = np.matrix(np.linalg.inv(S0))
    SN = np.linalg.pinv(S0_inv + beta * Phi.transpose() * Phi)
    mN = SN * (S0_inv * m0 + beta * Phi.transpose() * t)
    return np.array(mN).flatten(), np.array(SN)


def predictive_distribution(designmtx, beta, mN, SN):
    """
    Calculates the predictive distribution a linear model. This amounts to a
    mean and variance for each input point.

    parameters
    ----------
    designmtx - 2d (N x K) array of inputs (data-matrix or design-matrix) where
        N is the number of data-points and each row is that point's
        representation
    beta - the known noise precision
    mN - posterior mean of the weights (vector) 1d-array (or array-like)
        of length K
    SN - the posterior covariance matrix for the weights 2d (K x K)-array

    returns
    -------
    ys - a vector of mean predictions, one for each input datapoint
    sigma2Ns - a vector of variances, one for each input data-point
    """
    N, K = designmtx.shape
    Phi = np.matrix(designmtx)
    mN = np.matrix(mN).reshape((K, 1))
    SN = np.matrix(SN)
    ys = Phi * mN
    # create an array of the right size with the uniform term
    sigma2Ns = np.ones(N) / beta
    for n in range(N):
        # now calculate and add in the data dependent term
        # NOTE: I couldn't work out a neat way of doing this without a for-loop
        # NOTE: but if anyone can please share the answer.
        phi_n = Phi[n, :].transpose()
        sigma2Ns[n] += phi_n.transpose() * SN * phi_n
    return np.array(ys).flatten(), np.array(sigma2Ns)


def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx) * np.matrix(weights).reshape((len(weights), 1))
    return np.array(ys).flatten()


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
        # print(np.array(expanded_xs).shape)
        # print(np.array(ys).shape)
        return np.array(ys).flatten()

    # we return the function reference (handle) itself. This can be used like
    # any other function
    return prediction_function


def construct_3dpoly_bayesian(degree, weights, beta, SN):
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
        sigma2Ns = predictive_dist(beta, expanded_xs, SN)
        return np.array(ys).flatten(), sigma2Ns

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
        for i in range(degree + 1):
            expanded_inputs.append(col ** i)
    return np.array(expanded_inputs).transpose()


if __name__ == '__main__':
    main()