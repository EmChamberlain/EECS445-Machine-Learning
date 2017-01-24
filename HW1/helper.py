#!/usr/bin/env python

import numpy as np
import pandas as pd


def load_data(fname):
    '''
    Loads the data in file specified by fname. The file specified has to be a csv with first
    column being the label/output

    Returns X: an nx(d-1) array, where n is the number of examples and d is the dimensionality.
            Y: an nx1 array, where n is the number of examples
    '''

    data = pd.read_csv(fname).values
    X = data[:, 1:]
    y = data[:, 0]
    return X, y
