import knn_regression, polynomial_model, polynomial_model_bayesian, linear_regression, import_data


def main(datafile="datafile.csv"):
    wine_data, wine_features = import_data.import_data(datafile)
    linear_regression.main(wine_data, wine_features)
    knn_regression.main(wine_data)
    polynomial_model.main(wine_data)
    polynomial_model_bayesian.main(wine_data)


if __name__ == '__main__':
    import sys
    # this allows you to pass the file name as the first argument when you call
    # your script from the command line
    # so to run this script use:
    # python old_faithful.py old_faithful.tsv
    try:
        if len(sys.argv) < 0:
            main(sys.agrv[1])
        else:
            print("No File Passed, using datafile.csv ...")
            main()
    except IndexError:
        print("No valid file. Using backup data...")
        main("winequality-red.csv")