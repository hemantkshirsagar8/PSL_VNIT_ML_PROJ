import pandas as pd  #for manipulating data
import numpy as np  #for manipulating data
import sklearn  #for building models
import sklearn.ensemble  #for building models
from sklearn.model_selection import train_test_split  #for creating a hold-out sample
import shap  #SHAP package
import time  #some of the routines take a while, so we monitor the time
import os  #needed to use Environment Variables in Domino
import matplotlib.pyplot as plt  #for custom graphs at the end
import seaborn as sns  #for custom graphs at the end
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

X,y = shap.datasets.boston()
#print(X)
#print(y)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X.head()
#print(X.head())
pd.Series(y).head()
#print(pd.Series(y).head())
#print(sklearn.datasets.load_boston().DESCR)
#Random Forest
rf = sklearn.ensemble.RandomForestRegressor()
rf.fit(X_train, y_train)
#Tree on Random Forest:: we use tree explainer
explainerRF = shap.TreeExplainer(rf)
shap_values_RF_test = explainerRF.shap_values(X_test)
shap_values_RF_train = explainerRF.shap_values(X_train)
# Random Forest
#df_shap_RF_test = pd.DataFrame(shap_values_RF_test, columns=X_test.columns.values)
#df_shap_RF_train = pd.DataFrame(shap_values_RF_train, columns=X_train.columns.values)

#Pick an instance to explain
j = np.random.randint(0, X_test.shape[0])
#print(j)
j = 0
shap.initjs()

#SHAP
shap.force_plot(explainerRF.expected_value, shap_values_RF_test[j], X_test.iloc[[j]])
#shap.summary_plot(shap_values_RF_test,X_test)
#shap.summary_plot(shap_values_RF_train, X_train, plot_type="bar")
#shap.summary_plot(shap_values_RF_train, X_train)
#shp_plt = shap.dependence_plot("LSTAT", shap_values_RF_train, X_train)