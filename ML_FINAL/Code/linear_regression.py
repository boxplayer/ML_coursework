import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import polyfit, polyval, linalg
import scipy.stats as stats
from import_data import *


def main(wine_data, wine_features):
    print("Linear Regression")
    linear_regression_plot_together(wine_data, wine_features)

    # linear_regression_plot()
    # linear_regression_loop()
    # linear_regression_plot_all()
    # linear_regression_model()
    # linear_regression_regularised_model()
    # poly_model()
    # poly_model_reg()


def linear_regression_plot_together(wine_data, wine_features):
    """
    Plots the linear regression relationship of features and the label
    on one plot
    :return:
    """
    r_values = []
    y = wine_data[:, 11]
    wine_features = wine_features
    array = [1, 3, 5, 8, 9, 10]
    features = []
    for i in range(0, 6):
       features.append(wine_features[array[i]])
    wine_features = features
    wine_data = wine_data[:, (1, 3, 5, 8, 9, 10)]
    axes = []
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(16.5, 10.5)
    for i, a in enumerate(ax.flatten()):
        slope, intercept, r_value, p_value, std_err = stats.linregress(wine_data[:, i], y)
        r_values.append(r_value)
        x = wine_data[:, i]
        data, = a.plot(x, y, 'x')
        correlation, = a.plot(x, intercept + slope * x, 'r')
        a.legend(
            [data, correlation],
            ["Data",
             "Fitted Line $r^2$: %f" % r_value])
        a.set_xlabel(wine_features[i])
        a.set_ylabel("Quality")
        axes.append(a)
    fig.tight_layout()
    plt.show()
    fig.savefig("graphs/lin_reg.pdf", fmt="pdf", bbox_inches='tight')

    print("R values: ", r_values)
    return r_values

def linear_regression_plot():
    """
    Plots an individual linear regression graph
    :return:
    """
    wine_data = import_wine_data('datafile.csv')
    fixed_acidity = wine_data[:, 10]
    quality = wine_data[:, 11]
    print('Rows: %r' % (wine_data.shape[0]))
    print('Cols: %r' % (wine_data.shape[1]))
    slope, intercept, r_value, p_value, std_err = stats.linregress(fixed_acidity, quality)
    rsquared = r_value ** 2
    min_data = min(fixed_acidity)
    max_data = max(fixed_acidity)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = fixed_acidity
    y = quality
    data, = ax.plot(x, y, 'x')
    correlation, = ax.plot(x, intercept + slope * x, 'r')
    ax.legend(
        [data, correlation],
        ["Data",
         "Fitted Line $r^2$: %f" % r_value])
    ax.set_xlabel("acidity")
    ax.set_ylabel("quality")
    fig.tight_layout()
    plt.show()


def linear_regression_loop_plot_all():
    """
    Creates an individual plots of all the features against the label
    :return:
    """
    wine_data, wine_features = import_wine_data('datafile.csv')

    r_values = []
    y = wine_data[:, 11]
    for i in range(0, 11):
        slope, intercept, r_value, p_value, std_err = stats.linregress(wine_data[:, i], wine_data[:, 11])
        r_values.append(r_value)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = wine_data[:, i]
        data, = ax.plot(x, y, 'x')
        correlation, = ax.plot(x, intercept + slope * x, 'r')
        ax.legend(
            [data, correlation],
            ["Data",
             "Fitted Line $r^2$: %f" % r_value])
        ax.set_xlabel(wine_features[i])
        ax.set_ylabel("Quality")
        fig.tight_layout()
        plt.show()
    print(r_values)

def linear_regression_model():
    """
    Plots a simple mulitple linear regression graph comparing test to regression
    prediction and error of prediction
    """
    wine_data, wine_features = import_wine_data('datafile.csv')
    inputmtx = wine_data[:, 0:10]
    targets = wine_data[:, 11]
    weights = ml_weights(inputmtx, targets)
    predic = linear_model_predict(inputmtx, weights)
    num_samp = len(targets)
    x_name = np.linspace(1, num_samp + 1, num=num_samp)
    error = targets - predic
    error_per = abs(error*100/predic)
    mean_error = np.mean(error_per)
    mean_error_y = np.full((num_samp, 1), mean_error)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    prediction, = ax.plot(x_name, predic,'x')
    actual, = ax.plot(x_name, targets, 'or')
    ax.legend(
        [prediction, actual],
        ["Prediction",
         "Actual"])
    ax.set_xlabel("Sample")
    ax.set_ylabel("quality")
    fig.tight_layout()
    plt.show()
    fig.savefig("lin_reg_model.pdf", fmt="pdf", bbox_inches='tight')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    error, = ax2.plot(x_name, error_per, 'xg')
    mean, = ax2.plot(x_name, mean_error_y, 'r')
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("error %")
    ax2.legend(
        [error, mean],
        ["Error",
         "Mean error %f" % mean_error])
    fig2.tight_layout()
    plt.show()
    fig2.savefig("lin_reg_model_error.pdf", fmt="pdf", bbox_inches='tight')


def linear_regression_regularised_model():
    """
    Loops through a range of different regression coefficents to find the opitimun
    then plots the error and comparision graphs
    """
    wine_data, wine_features = import_wine_data('datafile.csv')
    inputmtx = wine_data[:, 0:10]
    targets = wine_data[:, 11]
    num_samp = len(targets)
    x_name = np.linspace(1, num_samp + 1, num=num_samp)
    correlation = []
    reg_coeffs = np.linspace(0, 1, 101)
    print(reg_coeffs)

    # Method for calculating the optimum regression coefficent
    for reg_coeff in reg_coeffs:
        weights = regulaised_ml_weights(inputmtx, targets, reg_coeff)
        predic = linear_model_predict(inputmtx, weights)
        error = targets - predic
        error_per = abs(error * 100 / predic)
        mean_error = np.mean(error_per)
        correlation.append(mean_error)

    # Get the best regression coefficient
    min_correlation = min(correlation)
    index_min = correlation.index(min_correlation)
    min_error_coeff = reg_coeffs[index_min]

    weights_opt = regulaised_ml_weights(inputmtx, targets, min_error_coeff)
    predic_opt = linear_model_predict(inputmtx, weights_opt)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # Plot the regularised coefficents
    # ax.plot(reg_coeffs, correlation, 'x')
    # Plot the Error
    prediction, = ax.plot(x_name, predic_opt, 'x')
    actual, = ax.plot(x_name, targets, 'or')
    ax.legend(
        [prediction, actual],
        ["Prediction",
         "Actual"])
    ax.set_xlabel("Sample")
    ax.set_ylabel("quality")
    fig.tight_layout()
    plt.show()
    fig.savefig("lin_reg_regularised_model.pdf", fmt="pdf", bbox_inches='tight')


    # Calulate and plot the new error
    error = targets - predic_opt
    error_per = abs(error * 100 / predic_opt)
    mean_error = np.mean(error_per)
    mean_error_y = np.full((num_samp, 1), mean_error)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    error, = ax2.plot(x_name, error_per, 'xg')
    mean, = ax2.plot(x_name, mean_error_y, 'r')
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("error %")
    ax2.legend(
        [error, mean],
        ["Error",
         "Mean error %f" % mean_error])
    fig2.tight_layout()
    plt.show()
    fig2.savefig("lin_reg_regularised_model_error.pdf", fmt="pdf", bbox_inches='tight')


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


def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()


def regulaised_ml_weights(inputmtx, targets, reg_coeff):
    """
    This method returns the regularised weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets),1))
    I = np.identity(Phi.shape[1])
    weights = linalg.pinv(I*reg_coeff + Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()


def poly_model():
    wine_data, wine_features = import_wine_data('datafile.csv')
    inputmtx = wine_data[:, 0:11]
    targets = wine_data[:, 11]
    degree = 3
    degrees = []
    rsme_arr = []
    for degree in range (0, 21):
        outputmtx = expand_to_2Dmonomials(inputmtx, degree)
        weights = ml_weights(outputmtx, targets)
        prediction_func = construct_3dpoly(degree, weights)
        prediction_values = prediction_func(inputmtx)
        rmse = np.sqrt(sum((targets - prediction_values) ** 2) / len(targets))
        rsme_arr.append(rmse)
        degrees.append(degree)

    # degrees = np.linspace(1, 20, 20)
    # print(degrees.size)
    # print(rsme_arr.size)
    min_correlation = min(rsme_arr)
    index_min = rsme_arr.index(min_correlation)
    min_error_deg = degrees[index_min]
    print(min_error_deg)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(degrees, rsme_arr, 'x')
    ax.set_xlabel("Degrees")
    ax.set_ylabel("RootMSE")
    fig.tight_layout()
    plt.show()



def poly_model_reg():
    """
    Loops through a range of different regression coefficents to find the opitimun
    then plots the error and comparision graphs
    """
    wine_data, wine_features = import_wine_data('datafile.csv')
    inputmtx = wine_data[:, 0:10]
    targets = wine_data[:, 11]
    num_samp = len(targets)
    correlation = []
    reg_coeffs = np.linspace(0, 1, 101)

    degrees = []
    rsme_arr = []
    for degree in range(0, 21):
        outputmtx = expand_to_2Dmonomials(inputmtx, degree)
        weights = ml_weights(outputmtx, targets)
        prediction_func = construct_3dpoly(degree, weights)
        prediction_values = prediction_func(inputmtx)
        rmse = np.sqrt(sum((targets - prediction_values) ** 2) / len(targets))
        rsme_arr.append(rmse)
        degrees.append(degree)

    # degrees = np.linspace(1, 20, 20)
    # print(degrees.size)
    # print(rsme_arr.size)
    min_correlation = min(rsme_arr)
    index_min = rsme_arr.index(min_correlation)
    min_error_deg = degrees[index_min]


    rsme_arr_coeff = []
    # Method for calculating the optimum regression coefficent
    for reg_coeff in reg_coeffs:
        outputmtx = expand_to_2Dmonomials(inputmtx, min_error_deg)
        weights = regulaised_ml_weights(outputmtx, targets, reg_coeff)
        prediction_func = construct_3dpoly(min_error_deg, weights)
        prediction_values = prediction_func(inputmtx)
        rmse = np.sqrt(sum((targets - prediction_values) ** 2) / len(targets))
        rsme_arr_coeff.append(rmse)

    print(rsme_arr_coeff)
    # Get the best regression coefficient
    min_rc = min(rsme_arr_coeff)
    index_min_rc = rsme_arr_coeff.index(min_rc)
    min_error_reg = reg_coeffs[index_min_rc]
    print(min_error_reg)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(reg_coeffs, rsme_arr_coeff, 'x')
    ax.set_xlabel("Coeffs")
    ax.set_ylabel("RootMSE")
    fig.tight_layout()
    plt.show()



def plot_function_and_data(inputs, targets, true_func, markersize=5):
    """
    Plot a function and some associated regression data in a given range

    parameters
    ----------
    inputs - the input data
    targets - the targets
    true_func - the function to plot
    markersize (optional) - the size of the markers in the plotted data
    <for other optional arguments see plot_function>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
    fig, ax, lines = plot_function(true_func)
    line, = ax.plot(inputs, targets, 'bo', markersize=markersize)
    lines.append(line)
    return fig, ax, lines


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
        for i in range(degree+1):
            expanded_inputs.append(col**i)
    return np.array(expanded_inputs).transpose()


if __name__ == '__main__':

  main()
