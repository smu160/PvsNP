#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:14:44 2018

@author: saveliyyusufov
"""

def isDataConsistent(python_DS, MATLAB_DS):
    """
    Checks two dataframes for inconsistencies
    """
    inconsistency = False
    for i in range(0, python_DS.columns.size):
        for j in range(0, python_DS[0].size):
            if int(python_DS[i][j]) != int(MATLAB_DS[i][j]):
                inconsistency = True
                print("Inconsistency found at: ({}, {}) ... python_DS[i][j] = {}, test[i][j] = {}".format(i, j, int(python_DS[i][j]), int(MATLAB_DS[i][j])))

    if inconsistency == False:
        return "No inconsistencies found"
    else:
        return "Inconsistencies found!"
    
    