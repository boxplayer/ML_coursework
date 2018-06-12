import csv

import numpy as np


def import_data(ifname):
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=';')
        # we want to avoid importing the header line.
        # instead we'll print it to screen
        header = next(datareader)
        # print("Importing data with fields:\n\t" + ",".join(header))
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
        # print('Rows: %r' % (data_as_array.shape[0]))
        # print('Cols: %r' % (data_as_array.shape[1]))
        # return this array to caller
        return data_as_array, header