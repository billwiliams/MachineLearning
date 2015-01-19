"""

%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.


% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
"""
#function to plot data
def plotData(X,y):
    """
    * function plotData takes two arguments X and y where X is the number of features and y the binary result and plots the data on a scatter plot


    """
    #importing numpy and pylab for plotting the data
    import numpy as np
    import pylab as pl
    #get the indices of the positive examples
    pos=np.where(y==1)[0]
    #get the indices of the negative examples
    neg=np.where(y==0)[0]
    #start pylab in interactive mode
    pl.ion();
    #create a figure
    fig=pl.figure();
    #plot data whose output is positive i.e Admitted
    pl.plot(X[pos, 0], X[pos, 1], 'k+',linewidth=2.0,c='b',label="admitted")
    #plot Data whose result i.e. y==0
    pl.plot(X[neg, 0], X[neg, 1], 'ko',c='y',label="Not admitted")
    
