import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

from Models.cross_validation import cross_validation
from Models.error_score import *
from Models.import_data import import_data


def main():
    rbf_scales()
    # rbf_centres()


def rbf_centres():
    wine_data, wine_features = import_data('winequality-red.csv')
    inputmtx = wine_data[:, 0:11]
    targets = wine_data[:, 11]
    train_inputs, train_targets, test_inputs, test_targets = cross_validation(inputmtx, targets, 0.25)
    scale = 2831 #Fix Scale
    scales = np.logspace(3, -2)
    centres, sd = classify_inputs(train_inputs, train_targets)
    # centres = [8.319637273,	0.527820513,	0.27097561,	2.538805503,	0.087466542,	15.87492183,	46.46779237,	0.996746679,	3.311113196,	0.658148843,	10.42298311,	5.636022514]
    centres = np.array(centres)
    feature_mapping = construct_rbf_feature_mapping(centres, scale)
    designmtx = feature_mapping(train_inputs)
    weights = ml_weights(designmtx, train_targets)
    rbf_approx = construct_feature_mapping_approx(feature_mapping, weights)
    prediction_values = rbf_approx(test_inputs)
    rmse, mae, medae, mape, variance = error_score(test_targets, prediction_values)

    print("Best RSME: ", rmse)


def classify_inputs(inputs, targets):
    mintarg = int(min(targets))
    maxtarg = int(max(targets))
    t = {}
    f = {}
    m = {}
    v = {}
    centres = np.zeros((inputs.shape[0], inputs.shape[1]))
    sd = np.zeros((inputs.shape[0], inputs.shape[1]))

    for n in range(mintarg, maxtarg + 1):
        t["group{0}".format(n)] = []
        f["group{0}".format(n)] = []
        m["group{0}".format(n)] = []
        v["group{0}".format(n)] = []

    for i in enumerate(targets):
        for n in range(mintarg, maxtarg + 1):
            if int(i[1]) == n:
                t["group{0}".format(n)].append(i)
                f["group{0}".format(n)].append(inputs[i[0], :])
                centres[i[0], :] = int(n)
                sd[i[0], :] = int(n)

    for group in f:
        array = np.array(f[group])
        for j in array.T:
            m[group].append(sum(j) / len(j))
            v[group].append(np.var(j))

    for i in range(0, centres.shape[0]):
        for j in range(0, centres.shape[1]):
            ind = int(centres[i, j])
            val = m["group{0}".format(ind)][j]
            var = v["group{0}".format(ind)][j]
            centres[i, j] = val
            sd[i, j] = var

    return centres, sd

def rbf_scales():
    wine_data, wine_features = import_data('winequality-red.csv')
    inputmtx = wine_data[:, 0:11]
    targets = wine_data[:, 11]
    inputs, targets, test_inputs, test_targets = cross_validation(inputmtx, targets, 0.25)

    # specify the centres of the rbf basis functions
    # centres = np.linspace(0, 1, 10)
    # centres, sd = classify_inputs(inputs, targets)
    centres = np.array([8.319637273,	0.527820513,	0.27097561,	2.538805503,	0.087466542,	15.87492183,	46.46779237,	0.996746679,	3.311113196,	0.658148843,	10.42298311,	5.636022514])
    # scales = np.logspace(4, 0)
    scales = np.linspace(100, 300, 200)
    errors = []
    for scale in scales:
        # print(scale)
    # define the feature mapping for the data
        feature_mapping = construct_rbf_feature_mapping(centres, scale)
        designmtx = feature_mapping(inputs)
        weights = ml_weights(designmtx, targets)
        rbf_approx = construct_feature_mapping_approx(feature_mapping, weights)
        prediction_values = rbf_approx(test_inputs)
        rmse, mae, medae, mape, variance = error_score(test_targets, prediction_values)
        errors.append(rmse)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(scales, errors, 'r')
    ax.set_xlabel("Scale")
    ax.set_ylabel("rmse")
    plt.show()

    min_rmse, min_scale = minimise_arr_error(errors, scales)
    print("Best RSME: ", min_rmse)
    print("Best Scale: ", min_scale)


def construct_rbf_feature_mapping(centres, scale):
    """
    parameters
    ----------
    centres - a DxM matrix (numpy array) where D is the dimension of the space
        and each row is the central position of an rbf basis function.
        For D=1 can pass an M-vector (numpy array).
    scale - a float determining the width of the distribution. Equivalent role
        to the standard deviation in the Gaussian distribution.

    returns
    -------
    feature_mapping - a function which takes an NxD data matrix and returns
        the design matrix (NxM matrix of features)
    """
    #  to enable python's broadcasting capability we need the centres
    # array as a 1xDxM array
    if len(centres.shape) == 1:
        centres = centres.reshape((1, 1, centres.size))
    else:
        centres = centres.reshape((1, centres.shape[1], centres.shape[0]))
    # the denominator
    denom = 2*scale**2
    # now create a function based on these basis functions
    def feature_mapping(datamtx):
        #  to enable python's broadcasting capability we need the datamtx array
        # as a NxDx1 array
        if len(datamtx.shape) == 1:
            # if the datamtx is just an array of scalars, turn this into
            # a Nx1x1 array
            datamtx = datamtx.reshape((datamtx.size, 1, 1))
        else:
            # if datamtx is NxD array, then we reshape matrix as a
            # NxDx1 array
            datamtx = datamtx.reshape((datamtx.shape[0], datamtx.shape[1], 1))
        sub = datamtx - centres
        summ = -np.sum((sub) ** 2, 2)
        calc = np.exp(summ/denom)
        return calc
    # return the created function
    return feature_mapping


def construct_feature_mapping_approx(feature_mapping, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        designmtx = np.matrix(feature_mapping(xs))
        return linear_model_predict(designmtx, weights)
    # we return the function reference (handle) itself. This can be used like
    # any other function
    return prediction_function


def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()


def ml_weights(inputmtx, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    weights = linalg.pinv(Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()

if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    main()