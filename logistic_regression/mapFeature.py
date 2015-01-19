"""
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size




"""
degree = 6;
import numpy as np

def mapFeature(X1,X2):
    ons = np.ones(np.size(X1[:]));
    ons=ons.reshape(np.size(X1[:]),1)
    out=np.ones(np.size(X1[:]))
    out=out.reshape(np.size(X1[:]),1)
    
    for j in range (1,degree+1):
        for i in range (j+1):
            out=np.column_stack((out,(X1 ** (j-i)) * (X2** i)))

    
    return out
