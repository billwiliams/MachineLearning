#a logistic regression model for predicting house prices

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import plotData
import mapFeature as mF


# Load Data
data = np.loadtxt(('ex2data2.txt'),delimiter=",");
X = data[:, 0:2]
y = data[:, 2:3].astype(int)
#mapping Features
X_map = mF.mapFeature(X[:,0], X[:,1]);
X_test=X_map[0:50,0:50]
y_test=y[0:50]

#plotting the data
plotData.plotData(X,y)

# Create logistic regression object. We dont fit an intercept term since MapFeature already does that for us
logreg = linear_model.LogisticRegression( fit_intercept=False)

# Train the model using the training sets
logreg.fit(X_map,y.ravel())
# The coefficients
print('Coefficients: \n', logreg.coef_)
h=.02
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % logreg.score(X_map, y))

# Plot the decision boundary
# point in the mesh [x_min, m_max]x[y_min, y_max].

u =np.linspace(-1, 1.5, 50);
u=u.reshape(np.size(u),1);
v = np.linspace(-1, 1.5, 50);
v=v.reshape(np.size(v),1);

z = np.matrix(np.zeros((len(u), len(v)),dtype=float));
m,n=np.shape(logreg.coef_)
for i in range (len(u)):
 for j in range(len(v)):
     z[i,j]=np.dot(mF.mapFeature(u[i], v[j]),logreg.coef_.reshape(n,m))

#reshaping back to original way to enable the plotting
u =np.linspace(-1, 1.5, 50);
v =np.linspace(-1, 1.5, 50);
#plotting a contour for the decision boundary z is transposed
plt.contour(u, v,np.transpose( z),(0,0),label="decision")



