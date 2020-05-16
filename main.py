# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:23:05 2020

@author: jhost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def viewData(data):
    print('\nWould you like to view the first few rows of the data? [ Y / N ]')
    decision = input().upper()
    if decision == 'Y':
        print('\nHow many rows would you like to view from the data? (Enter an integer value.)')
        rows = int(input())
        print(data.head(rows))

def changeName(data):
    changeName = True
    while changeName:
        print('\nAre there any column names you want to change? If yes, type in a single column name, otherwise, type \"CANCEL\".')
        old_col_name = input()
        
        if old_col_name == 'CANCEL':
            changeName = False
            break
            
        print('\nWhat would you like to change the column \"%s\" to? Type in a single column name, otherwise, type \"CANCEL\".' % (old_col_name))
        new_col_name = input()
        
        if new_col_name == 'CANCEL':
            changeName = False
            break
        else:
            data.rename(columns={old_col_name : new_col_name}, inplace=True)
        viewData(data)
        
def assignDataTypes(data):
    global data_types
    print('\nPlease assign the data type to each column.')
    for col in data.columns:
        print('\nWhat data type is column \"%s\"? (Type \"CANCEL\" to return to menu.)' % (col)) 
        print('Options: [ 0 -> Nominal, 1 -> Ordinal, 2 -> Interval, 3-> Ratio ]')
        print('\nExample rows:')
        print(data[col].iloc[0:10])
        entry = input()
        
        if entry.upper() == 'CANCEL':
            break
        
        data_type_int = int(entry)
        
        if data_type_int == 0:
            data_type = 'Nominal'
        elif data_type_int == 1:
            data_type = 'Ordinal'
        elif data_type_int == 2:
            data_type = 'Interval'
        elif data_type_int == 3:
            data_type = 'Ratio'
        else:
            print('Unrecognized entry: \"%s\"' % (data_type_int))
            
        data_types[col] = data_type
        
def missingData(data):
    if data.isnull().any().any():
        print('\nMissing data has been identified.')
        print('\nWhich interpolation method would you like to perform? (Type in the name of the method.)')
        print('Options: [ \"linear\", \"time\", \"index\", \"values\", \"nearest\", \"zero\", \"slinear\", '
              + '\"quadratic\", \"cubic\", \"barycentric\", \"krogh\", \"polynomial\", \"spline\", '
              + '\"piecewise_polynomial\", \"from_derivatives\", \"pchip\", \"akima\" ]')
        interpolate_method = input().lower()
        data.interpolate(method=interpolate_method, limit_direction='both', inplace=True)
        print('\nInterpolation Finished. All missing data has been resolved.')
        viewData(data)
    else:
        print('\nNo missing data has been identified.')
        
def covariance(data):
    print(data.cov())

def correlation(data):
    print(data.corr())
    print('\nRemember: an absolute value of a correlation entry of 0.6 or greater is significant.')
    
def boxAndWhiskerPlot(data):
    print('\nWhich attribute do you want to draw a box and whisker plot of? (Type \"ALL\" to draw a box and whisker plot for each attribute.')
    print('Options: %s' % (list(data.columns)))
    selection = input()
    if selection.upper() == 'ALL':
        for column in data.columns:
            plt.boxplot(data[column], vert=False, showfliers=True)
            plt.title(column)
            plt.show()
    elif selection in list(data.columns):
        plt.boxplot(data[selection], vert=False, showfliers=True)
        plt.title(selection)
        plt.show()
    else:
        print('Invalid attribute.')
        
def drawHistogram(data, attribute):
    n, bins, patches = plt.hist(x=data[attribute], bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('%s' % (attribute))
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
        
def histogram(data):
    print('\nWhich attribute do you want to draw a histogram of?'
          + '(Type \"ALL\" to draw a box and whisker plot for each attribute.')
    print('Options: %s' % (list(data.columns)))
    selection = input()
    if selection.upper() == 'ALL':
        for column in data.columns:
            drawHistogram(data, column)
    elif selection in list(data.columns):
        drawHistogram(data, selection)
    else:
        print('Invalid attribute.')
        
def stdSummaryStatistics(data):
    print('\nWhich attribute do you want to calculate standard summary statistics for?'
          + '(Type \"ALL\" to draw a box and whisker plot for each attribute.')
    print('Options: %s' % (list(data.columns)))
    selection = input()
    if selection.upper() == 'ALL':
        for column in data.columns:
            print('%s:' % (column))
            print('Mean = %s' % (data[column].mean()))
            print('Median = %s' % (data[column].median()))
            print('Mode = %s' % (data[column].mode()))
            print('Range = %s' % (data[column].max() - data[column].min()))
            print('Standard Deviation = %s' % (data[column].std()))
            print('Variance = %s' % (data[column].var()))
            
    elif selection in list(data.columns):
        plt.boxplot(data[selection], vert=False, showfliers=True)
        plt.title(selection)
        plt.show()
    else:
        print('Invalid attribute.')
        
def normalization(data):    
    scaler = MinMaxScaler() 
    scaled_values = scaler.fit_transform(data.iloc[:,1:]) 
    data.loc[:,1:] = scaled_values
    
def standardization(data):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(data.iloc[:,1:])
    data.loc[:,1:] = scaled_values
                
def menu(data):
    while True:
        options = ['View data', 'Change data attribute names', 'Assign data types', 
                   'Check and/or resolve missing data', 'Covariance matrix', 'Correlation matrix',
                   'Box and Whisker Plot', 'Normalization', 'Standardization', 'Histogram']
        switcher = {
            1: viewData,
            2: changeName,
            3: assignDataTypes,
            4: missingData,
            5: covariance,
            6: correlation,
            7: boxAndWhiskerPlot,
            8: normalization,
            9: standardization,
            10: histogram,
        }
        
        # display menu
        print()
        print('-'*20 + ' Menu ' + '-'*20)
        for i in range(len(options)):
            print('%s : %s' % (i + 1, options[i]))
        
        selection = int(input())
        
        # Get the function from switcher dictionary
        func = switcher.get(selection, lambda: "Invalid selection.")
        
        # Execute the function
        func(data)
        
data_types = {}
execute = True
while execute:
    print('Enter the file location:')
    filename = input()
    test = 'data/Advertising.csv'
    #filename = test
    print('\nLocating file...')
    # read in the provided data
    try:    
        execute = False
        data = pandas.read_csv(filename)
        print('File found.')
        print('\nLoading data...')
        
        if data.empty:
            print('The data is empty.')
        else:
            print('Data retrieved.')
            menu(data)

    except FileNotFoundError:
        print('File was not found.')