from Models import knn_regression, PolynomialModel, polynomial_mode_bayesian_cv, linear_regression


def main():
    linear_regression.main()
    knn_regression.main()
    PolynomialModel.main()
    polynomial_mode_bayesian_cv.main()


if __name__ == '__main__':
    main()