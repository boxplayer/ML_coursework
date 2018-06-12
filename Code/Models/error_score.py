import numpy as np


def error_score(targets, prediction_values):
    rmse = rms_error(targets, prediction_values)
    mae = mean_abs_error(targets, prediction_values)
    medae = median_abs_error(targets, prediction_values)
    mape = mean_abs_perc_error(targets, prediction_values)
    variance = variance_error(targets, prediction_values)

    return rmse, mae, medae, mape, variance


def rms_error(targets, prediction_values):
    rsme = np.sqrt(sum((targets - prediction_values) ** 2) / len(targets))
    return rsme


def mean_abs_error(targets, prediction_values):
    error = targets - prediction_values
    mae = np.mean(abs(error))
    return mae


def median_abs_error(targets, prediction_values):
    error = targets - prediction_values
    medae = np.median(abs(error))
    return medae


def mean_abs_perc_error(targets, prediction_values):
    error = targets - prediction_values
    error_per = abs(error * 100 / prediction_values)
    mape = np.mean(error_per)
    return mape


def variance_error(targets, prediction_values):
    error = targets - prediction_values
    variance = np.var(error)
    return variance


def minimise_mtx_error(errormtx, varied_para_i, varied_para_j):
    min_error = errormtx.min()
    min_index = np.unravel_index(errormtx.argmin(), errormtx.shape)
    min_i = varied_para_i[min_index[0]]
    min_j = varied_para_j[min_index[1]]
    return min_error, min_i, min_j


def minimise_mtx_2d_error(errormtx, varied_para_i):
    min_error = errormtx.min()
    min_index = np.unravel_index(errormtx.argmin(), errormtx.shape)
    min_i = varied_para_i[min_index[0]]
    return min_error, min_i


def minimise_arr_error(errorarr, varied_para):
    min_error = min(errorarr)
    min_index = errorarr.index(min_error)
    min_param = varied_para[min_index]
    return min_error, min_param


def print_errors_mtx(errormtx, varied_para_i, varied_para_j, error_name):
    min_error, min_i, min_j = minimise_mtx_error(errormtx, varied_para_i, varied_para_j)
    print("Best Regression Coefficient: ", min_i)
    print("Best Polynomial Degree: ", min_j)
    print("Best ", error_name, ": ", min_error, "\n")


def print_errors_3d_mtx(errormtx, varied_para_i, varied_para_j):
    error_name = ["RMS", "Mean", "Median", "Mean Percentage", "Variance"]
    for n in range(0, 5):
        min_error, min_i, min_j = minimise_mtx_error(errormtx[:, :, n], varied_para_i, varied_para_j)
        print(error_name[n])
        print("Best Regression Coefficient: ", min_i)
        print("Best Polynomial Degree: ", min_j)
        print("Best ", error_name[n], ": ", min_error, "\n")
    return min_i, min_j


def print_errors_2d_mtx(errormtx, varied_para_i):
    error_name = ["RMS", "Mean", "Median", "Mean Percentage", "Variance"]
    for n in range(0, 5):
        min_error, min_i = minimise_mtx_2d_error(errormtx[:, n], varied_para_i)
        print(error_name[n])
        print("Best K Value: ", min_i)
        print("Best ", error_name[n], ": ", min_error, "\n")


def print_error_score(errorarray):
    error_name = ["RMS", "Mean", "Median", "Mean Percentage", "Variance"]
    for n in range(0, 5):
        print("Best ", error_name[n], ": ", errorarray[n])
