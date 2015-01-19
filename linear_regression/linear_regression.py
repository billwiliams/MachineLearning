#a linear regression model for predicting house prices


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import Plotlearning as Pl
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB

# Load Data
data = np.loadtxt(('ex1data2.txt'),delimiter=",");
X = data[:, 0:2]
y = data[:, 2:3]
c=data[:, 2:3].flatten(0)

#training samples
X_train = X[0:46,0:2]
y_train=y[:46]
# testing sets
X_test=X[30:46,0:2]
y_test=y[30:46]

# Create linear regression object
regr = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True)

# Train the model using the training sets
regr.fit(X_train,y_train)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

# Plot outputs

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=100,
                                   test_size=0.2, random_state=0)

estimator = GaussianNB()
Pl.plot_learning_curve(estimator, title, X, c, ylim=None, cv=cv, n_jobs=4)
plt.show()

