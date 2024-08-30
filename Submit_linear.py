import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import time as tm


def my_fit(x,y):
        x_new = my_map(x)
        y_new = 2*y-1
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_new)

        # Set up the SGDClassifier with the same hyperparameters
        my_clf = LinearSVC(C = 100)

        # Fit the model
        my_clf.fit(x_scaled, y_new)

        # Extract the weights and bias
        weights = my_clf.coef_.reshape(-1)
        bias = my_clf.intercept_[0]

        # Return weights and bias
        return weights, bias

################################
# Non Editable Region Starting #
################################
def my_map( x ):
################################
#  Non Editable Region Ending  #
################################
    my_ones = np.ones((x.shape[0],1))
    x_new = np.hstack((x,my_ones))
    for j in range(31,-1,-1):
          x_new[:,j] = (1-2*x_new[:,j])*x_new[:,j+1]
    my_list = []
    for j in range(32):
          for k in range(j+1,32):
                my_list.append(x_new[:,j]*x_new[:,k])
    for j in range(32):
          my_list.append(x_new[:,j])
    return np.array(my_list).T