import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import polyfit, polyval, linalg
from regression_samples import arbitrary_function_1
from regression_models import expand_to_monomials
import scipy.stats as stats

def main():
    # linear_regression_plot()
    # linear_regression_loop()
    # linear_regression_plot_all()
    # linear_regression_plot_together()
    linear_regression_model()


def linear_regression_model():
    wine_data, wine_features = import_wine_data('winequality-red.csv')
    inputmtx = wine_data[:, 0:11]
    targets = wine_data[:, 11]

    """
    Split data into train and test 
    """
    test_fraction = 0.25
    N = inputmtx.shape[0]
    p = [test_fraction,(1-test_fraction)]
    train_part = np.random.choice([False,True],size=N, p=p)
    test_part = np.invert(train_part)
    train_inputmtx = inputmtx[train_part,:]
    test_inputmtx = inputmtx[test_part,:]
    train_targets = targets[train_part]
    test_targets = targets[test_part]

    """
    """

    weights = ml_weights(train_inputmtx, train_targets)
    predic = linear_model_predict(test_inputmtx, weights)
    num_samp = len(test_targets)
    x_name = np.linspace(1, num_samp + 1, num=num_samp)
    error = test_targets - predic;
    rmse = np.sqrt(sum((test_targets - predic) ** 2) / len(test_targets))
    print("Root mean squared error: %f" % rmse)

    # N = 20
    # degree = 2
    # processed_inputs = expand_to_monomials(inputmtx, degree)
    # print(inputmtx.shape)
    # print(processed_inputs.shape)
    # # weights2 = ml_weights(processed_inputs, targets)



    error_per = abs(error*100/predic)
    mean_error = np.mean(error_per);
    mean_error_y = np.full((num_samp, 1), mean_error)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    prediction, = ax.plot(x_name, predic,'x')
    actual, = ax.plot(x_name, test_targets, 'or')
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



    # print(weights)



def ml_weights(inputmtx, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    Phi = np.matrix(inputmtx)
    targets = np.matrix(targets).reshape((len(targets), 1))
    weights = linalg.inv(Phi.transpose() * Phi) * Phi.transpose() * targets
    return np.array(weights).flatten()

def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx)*np.matrix(weights).reshape((len(weights),1))
    return np.array(ys).flatten()

def import_wine_data(ifname):
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=';')
        # we want to avoid importing the header line.
        # instead we'll print it to screen
        header = next(datareader)
        print("Importing data with fields:\n\t" + ",".join(header))
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            # print("row = %r" % (row,))
            # for each row of data
            # convert each element (from string) to float type
            row_of_floats = list(map(float,row))
            # print("row_of_floats = %r" % (row_of_floats,))
            # now store in our data list
            data.append(row_of_floats)
        # convert the data (list object) into a numpy array.
        data_as_array = np.array(data)
        print('Rows: %r' % (data_as_array.shape[0]))
        print('Cols: %r' % (data_as_array.shape[1]))
        # return this array to caller
        return data_as_array, header

if __name__ == '__main__':

  main()