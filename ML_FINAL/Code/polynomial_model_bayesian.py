from numpy.linalg import linalg
import numpy as np
from error_score import *
from results_plots import *


def main(wine_data):
    print("Polynomial Bayesian Model")
    poly_model_reg(wine_data)

def poly_model_reg(wine_data):
    """
    Loops through a range of different regression coefficents to find the optimum
    then plots the error and comparison graphs
    """
    inputmtx = wine_data[:, 0:11]
    targets = wine_data[:, 11]

    # Create Folds
    num_folds = 10
    N = len(targets)
    folds = create_cv_folds(N, num_folds)
    errormtxs, uppers, lowers, prediction_values = cv_evaluation_poly_model(inputmtx, targets, folds, num_folds)

    preds_arry = np.zeros(160)
    upper_arr = np.zeros(160)
    lower_arr = np.zeros(160)
    for x in range (0,prediction_values.shape[1]):
        preds_arry[x] = np.mean(prediction_values[:, x])
        upper_arr[x] = np.mean(uppers[:, x])
        lower_arr[x] = np.mean(lowers[:, x])
    errormean = np.zeros((1, 5))
    for fold in errormtxs:
        matrix = errormtxs[fold]
        errormean = matrix + errormean


    errormean = errormean/num_folds

    errormean = errormean[0, :, :]
    it_priors = np.linspace(1, 50, 50)


    display_error_graphs(errormean, it_priors, "Number of Priors",
                         "Change in Priors", "priors")


    display_bayesian_confidence_graph(preds_arry, upper_arr, lower_arr)
    min_rsme_index = np.argmin(errormean, 0)[0]
    print_error_score(errormean[min_rsme_index])


def polynomial_bayesian(train_inputmtx, train_targets, test_inputmtx, test_targets, targets):


    min_poly_deg = 3
    min_regceof = 0.26
    outputmtx = expand_to_2Dmonomials(train_inputmtx, min_poly_deg)
    M = outputmtx.shape[1]
    m0 = np.repeat(0, M)
    alpha = 100


    """
    Deduce Beta
                   
    1. Via variance of raw targets 
    2. Via variance of residuals acquired by linear regression 
    """
    lin_weights = regularised_ml_weights(train_inputmtx, train_targets, min_regceof)
    prediction_values = linear_model_predict(test_inputmtx, lin_weights)
    resdiduals = test_targets - prediction_values
    resdidual_var = np.var(resdiduals)
    beta2 = (1. / resdidual_var) ** 2

    raw_var = np.var(targets)
    beta = (1. / raw_var) ** 2
    """"""
    S0 = alpha * np.identity(M)
    # mN, SN = calculate_weights_posterior(outputmtx, train_targets, beta, m0, S0)
    # prediction_func = construct_3dpoly(min_poly_deg, mN)
    # prediction_values = prediction_func(test_inputmtx)
    # errorarr = error_score(test_targets, prediction_values)
    mN, SN = calculate_weights_posterior(outputmtx, train_targets, beta, m0, S0)
    def calc_post(priorM, priorS, i, errorarr=[]):
        mn, sn = calculate_weights_posterior(outputmtx, train_targets, beta, priorM, priorS)
        prediction_func = construct_3dpoly(min_poly_deg, mn, beta)
        prediction_values, sigma2Ns = prediction_func(test_inputmtx, sn)
        errorarr.append(error_score(test_targets, prediction_values))

        if i < 49:
            i += 1
            mn, sn, errorarr = calc_post(mn, sn, i, errorarr)
        # err_bay_arr = np.delete(err_bay_arr, 0, axis=0)
        # print(err_bay_arr)
        return mn, sn, errorarr

    mn, sn, errorarr = calc_post(mN, SN, 0)

    prediction_func = construct_3dpoly(min_poly_deg, mn, beta)
    prediction_values, sigma2Ns = prediction_func(test_inputmtx, sn)
    lower = prediction_values - np.sqrt(sigma2Ns)
    upper = prediction_values + np.sqrt(sigma2Ns)
    return errorarr, lower, upper, prediction_values



def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights), 1))
    return np.array(ys).flatten()

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
    t = np.matrix(targets).reshape((N,1))
    m0 = np.matrix(m0).reshape((K,1))
    S0_inv = np.matrix(np.linalg.inv(S0))
    SN = np.linalg.pinv(S0_inv + beta*Phi.transpose()*Phi)
    mN = SN*(S0_inv*m0 + beta*Phi.transpose()*t)
    return np.array(mN).flatten(), np.array(SN)

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


def construct_3dpoly(degree, weights, beta):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """

    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs, SN):
        expanded_xs = np.matrix(expand_to_2Dmonomials(xs, degree))
        N, K = expanded_xs.shape
        Phi = np.matrix(expanded_xs)
        SN = np.matrix(SN)
        sigma2Ns = np.ones(N) / beta
        for n in range(N):
            # now calculate and add in the data dependent term
            # NOTE: I couldn't work out a neat way of doing this without a for-loop
            # NOTE: but if anyone can please share the answer.
            phi_n = Phi[n, :].transpose()
            sigma2Ns[n] += phi_n.transpose() * SN * phi_n
        ys = expanded_xs * np.matrix(weights).reshape((len(weights)), 1)
        return np.array(ys).flatten(), np.array(sigma2Ns)

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
            expanded_inputs.append(col ** i)
    return np.array(expanded_inputs).transpose()


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
        folds.append((train, test))
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


def cv_evaluation_poly_model(inputs, targets, folds, num_folds):
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
    errors = {}
    for f, fold in enumerate(folds):
        errors["fold{0}".format(f)] = []

    uppers = np.zeros((num_folds, 160))
    lowers = np.zeros((num_folds, 160))
    preds = np.zeros((num_folds, 160))
    for f, fold in enumerate(folds):
        # f is the fold id, fold is the train-test split
        train_part, test_part = fold
        # break the data into train and test sets
        train_inputs, train_targets, test_inputs, test_targets = \
            train_and_test_partition(inputs, targets, train_part, test_part)
        # now train and evaluate the error on both sets
        errorarray, lower, upper, prediction_values = polynomial_bayesian(train_inputs, train_targets, test_inputs, test_targets, targets)
        if (upper.size < 160):
            upper = upper.tolist()
            upper.append(upper[158])
            upper = np.array(upper)
            lower = lower.tolist()
            lower.append(lower[158])
            lower = np.array(lower)
            prediction_values = prediction_values.tolist()
            prediction_values.append(prediction_values[158])
            prediction_values = np.array(prediction_values)
        uppers[f] = upper
        lowers[f] = lower
        preds[f] = prediction_values
        errors["fold{0}".format(f)].append(errorarray)

    return errors, uppers, lowers, preds


if __name__ == '__main__':
    main()
