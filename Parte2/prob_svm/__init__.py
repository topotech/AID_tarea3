# *-* coding: utf-8 *-*
#Thanks to Eren Golge for his inspirational script.
#Gracias a Eren Golge por su código inspirador

from sklearn.svm import LinearSVC
import numpy as np

class LinearSVC_proba(LinearSVC):
    '''
    Original code:

    def __platt_func(self,x):
        return 1/(1+np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions)

        print platt_predictions
        probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        return probs


    '''

    def predict_proba(self, X):
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)

        prob += 1
        np.reciprocal(prob, prob)
        if len(self.classes_) == 2:  # binary case
            return np.column_stack([1 - prob, prob])
        else:
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob
