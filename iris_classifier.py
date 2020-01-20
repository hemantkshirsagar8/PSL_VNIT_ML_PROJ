
# coding: utf-8

# In[1]:

import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import time
import shap
from warnings import simplefilter
simplefilter(action = 'ignore', category = FutureWarning )


# In[2]:

X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(),test_size = 0.2,random_state = 0)
def print_accuracy(f):
    print("Accuracy = {0}%".format(100*np.sum(f(X_test) == Y_test)/len(Y_test)))
    time.sleep(0.5)
shap.initjs()


# In[9]:

linear_lr = sklearn.linear_model.LogisticRegression()
linear_lr.fit(X_train,Y_train)
print_accuracy(linear_lr.predict)
explainer = shap.KernelExplainer(linear_lr.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values,X_test)

