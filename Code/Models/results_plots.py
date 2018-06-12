from matplotlib import rcParams
import matplotlib
# matplotlib.font_manager._rebuild()
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Avenir Next']
import matplotlib.pyplot as plt

import numpy as np


def display_error_graphs(errors, variable,  variable_name, title, file_name):
    error_name = [r"${E}_{RMS}$", r"${ E }_{ M }$", r"${ E }_{ \tilde { x }  }$", r"${ E }_{ M \% }$", r"${ \sigma }^{ 2 }$"]
    error_namefile = ["RMS", "Mean", "Median", "MeanPercentage", "Variance"]
    for type in range(0, 5):
        error_arr = errors[:, type]
        min_var = min(variable)
        max_var = max(variable)
        min_err = min(np.array(error_arr))
        max_err = max(np.array(error_arr))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(variable, error_arr, 'b',  linewidth=1.5)
        ax.set_xlabel(variable_name, fontsize=12)
        ax.set_ylabel(error_name[type], fontsize=12)
        ax.set_xlim(min_var, max_var)
        ax.set_ylim(min_err, max_err)
        plt.title(error_name[type] + " vs " + title)
        fig.tight_layout()
        plt.grid(True)
        fig.savefig("../../MachineLearningReport/LatestBuild/Img/" + error_namefile[type] + "vs" + file_name + ".pdf", fmt="pdf")
        plt.show()


def display_3d_error_graphs(threemtx, reg_coeffs, degrees):
    error_name = ["RMS", "Mean", "Median", "Mean Percentage", "Variance"]
    for type in range(0, 5):
        error_arr = threemtx[:, :, type]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Degree")
        ax.set_ylabel(error_name[type])
        ax.set_ylim(0.3, 0.8)
        ax.set_xlim(0, 6)
        plt.title(error_name[type] + " vs Degree for each reg_coeff")
        for coef in range(0, 50):
            ax.plot(degrees, error_arr[coef, :], linewidth=0.1)
        fig.savefig("poly_graphs/coeff_" + error_name[type] + "vsDegree.pdf", fmt="pdf")
        plt.show()

def display_bayesian_confidence_graph(preds, uppers, lowers):
    # min_low = min(lowers)
    # max_up = max(uppers)
    # min_preds = min(preds)
    # max_preds = max(preds)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(preds, uppers, 'bx')
    ax.plot(preds, lowers, 'rx')
    ax.set_xlabel("Predicted Values", fontsize=12)
    ax.set_ylabel("Upper(blue) and Lower(red) Bounds of Predicted Values", fontsize=12)
    # ax.set_xlim(min_low, max_up)
    # ax.set_ylim(min_preds, max_preds)
    plt.title("Predictive Distribution of Bayesian Polynomial Model" )
    fig.tight_layout()
    plt.grid(True)
    fig.savefig("../../MachineLearningReport/LatestBuild/Img/predictive_distribution.pdf",
                fmt="pdf")
    plt.show()