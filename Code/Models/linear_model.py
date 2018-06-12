import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import linalg

from Models.cross_validation import cross_validation
from Models.import_data import import_data


def main():
    linear_model()


def linear_model():
    """
    Loops through a range of different regression coefficents to find the opitimun
    then plots the error and comparision graphs
    """
    wine_data, wine_features = import_data('winequality-red.csv')
    inputmtx = wine_data[:, 0:10]
    targets = wine_data[:, 11]
    train_inputmtx, train_targets, test_inputmtx, test_targets = \
        cross_validation(inputmtx, targets, 0.25)
    rsmes = []
    reg_coeffs = np.linspace(0, 1, 51)
    correlation = []

    # Method for calculating the optimum regression coefficent
    for reg_coeff in reg_coeffs:
        weights = regularised_ml_weights(inputmtx, targets, reg_coeff)
        prediction_values = linear_model_predict(inputmtx, weights)
        rmse = np.sqrt(sum((targets - prediction_values) ** 2) / len(targets))
        rsmes.append(rmse)
        error = targets - prediction_values
        error_per = abs(error * 100 / prediction_values)
        mean_error = np.mean(error_per)
        correlation.append(mean_error)

        # Get the best regression coefficient
    min_correlation = min(correlation)
    index_min = correlation.index(min_correlation)
    min_error_coeff_corr = reg_coeffs[index_min]

    # Get the best regression coefficient
    min_rc = min(rsmes)
    index_min_rc = rsmes.index(min_rc)
    min_error_coeff = reg_coeffs[index_min_rc]
    print(min_error_coeff)

    weights_opt = regularised_ml_weights(inputmtx, targets, min_error_coeff)
    predic_opt = linear_model_predict(inputmtx, weights_opt)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(reg_coeffs, correlation, 'x')
    ax.set_xlabel("Coeffs")
    ax.set_ylabel("error")
    fig.tight_layout()
    plt.show()


def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights), 1))
    return np.array(ys).flatten()


def regularised_ml_weights(inputmtx, targets, reg_coeff):
    """
    This method returns the regularised weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets), 1))
    I = np.identity(Phi.shape[1])
    weights = linalg.inv(I*reg_coeff + Phi.transpose()*Phi)*Phi.transpose()*targets
    return np.array(weights).flatten()


if __name__ == '__main__':

  main()
