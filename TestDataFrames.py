#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:14:44 2018

@author: Saveliy Yusufov, Columbia University, sy2685@columbia.edu
"""
import numba


@numba.jit
def isDataConsistent(python_DS, MATLAB_DS):
    """
    Checks two dataframes for inconsistencies
    """
    inconsistency = False
    for column in python_DS:
        for j in range(0, len(python_DS[column])):
            if int(python_DS[column][j]) != int(MATLAB_DS[column][j]):
                inconsistency = True
                print("Inconsistency found at: ({}, {}) ... python_DS[{}][{}] = {}, test[{}][{}] = {}".format(column, j, column, j, python_DS[column][j], column, j, MATLAB_DS[column][j]))

    if inconsistency is False:
        return "No inconsistencies found"
    else:
        return "Inconsistencies found!"
    
    