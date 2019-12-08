import pandas as pd  #for manipulating data
import numpy as np  #for manipulating data
import sklearn  #for building models
import xgboost as xgb  #for building models
import sklearn.ensemble  #for building models
from sklearn.model_selection import train_test_split  #for creating a hold-out sample
import lime  #LIME package
import lime.lime_tabular  #the type of LIIME analysis we’ll do
import shap  #SHAP package
import time  #some of the routines take a while, so we monitor the time
import os  #needed to use Environment Variables in Domino
import matplotlib.pyplot as plt  #for custom graphs at the end
import seaborn as sns  #for custom graphs at the end



'''print(shap.__version__)
print(xgb.__version__)
print(sklearn.__version__)'''



X,y = shap.datasets.boston()
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X.head()
pd.Series(y).head()
#print(sklearn.datasets.load_boston().DESCR)




#XGBoost
xgb_model = xgb.train({'objective':'reg:linear'}, xgb.DMatrix(X_train, label=y_train))


#GBT from scikit-learn
sk_xgb = sklearn.ensemble.GradientBoostingRegressor()
sk_xgb.fit(X_train, y_train)


#Random Forest
rf = sklearn.ensemble.RandomForestRegressor()
rf.fit(X_train, y_train)


#K Nearest Neighbor
knn = sklearn.neighbors.KNeighborsRegressor()
knn.fit(X_train, y_train)





#Tree on XGBoost
explainerXGB = shap.TreeExplainer(xgb_model)
shap_values_XGB_test = explainerXGB.shap_values(X_test)
shap_values_XGB_train = explainerXGB.shap_values(X_train)


#Tree on Scikit GBT
explainerSKGBT = shap.TreeExplainer(sk_xgb)
shap_values_SKGBT_test = explainerSKGBT.shap_values(X_test)
shap_values_SKGBT_train = explainerSKGBT.shap_values(X_train)


#Tree on Random Forest
explainerRF = shap.TreeExplainer(rf)
shap_values_RF_test = explainerRF.shap_values(X_test)
shap_values_RF_train = explainerRF.shap_values(X_train)




"""
Rather than use the whole training set to estimate expected values, we summarize with
a set of weighted kmeans, each weighted by the number of points they represent.
Running without the kmeans took 1 hr 6 mins 7 sec. 
Running with the kmeans took 2 min 47 sec.
Boston Housing is a very small dataset.
Running SHAP on models that require Kernel method and have a good amount of data becomes prohibitive. 
"""
X_train_summary = shap.kmeans(X_train, 10)


# using kmeans
t0 = time.time()
explainerKNN = shap.KernelExplainer(knn.predict, X_train_summary)
shap_values_KNN_test = explainerKNN.shap_values(X_test)
shap_values_KNN_train = explainerKNN.shap_values(X_train)
t1 = time.time()
timeit=t1-t0
timeit


# without kmeans a test run took 3967.6232330799103 seconds
"""
t0 = time.time()
explainerKNN = shap.KernelExplainer(knn.predict, X_train)
shap_values_KNN_test = explainerKNN.shap_values(X_test)
shap_values_KNN_train = explainerKNN.shap_values(X_train)
t1 = time.time()
timeit=t1-t0
timeit 
"""


#Get the SHAP values into dataframes so we can use them later on
# XGBoost
df_shap_XGB_test = pd.DataFrame(shap_values_XGB_test, columns=X_test.columns.values)
df_shap_XGB_train = pd.DataFrame(shap_values_XGB_train, columns=X_train.columns.values)


# Scikit GBT
df_shap_SKGBT_test = pd.DataFrame(shap_values_SKGBT_test, columns=X_test.columns.values)
df_shap_SKGBT_train = pd.DataFrame(shap_values_SKGBT_train, columns=X_train.columns.values)


# Random Forest
df_shap_RF_test = pd.DataFrame(shap_values_RF_test, columns=X_test.columns.values)
df_shap_RF_train = pd.DataFrame(shap_values_RF_train, columns=X_train.columns.values)


# KNN
df_shap_KNN_test = pd.DataFrame(shap_values_KNN_test, columns=X_test.columns.values)
df_shap_KNN_train = pd.DataFrame(shap_values_KNN_train, columns=X_train.columns.values)


#Creating the LIME Explainer
# if a feature has 10 or less unique values then treat it as categorical
categorical_features = np.argwhere(np.array([len(set(X_train.values[:,x]))for x in range(X_train.values.shape[1])]) <= 10).flatten()

# LIME has one explainer for all models
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,feature_names=X_train.columns.values.tolist(),class_names=['price'],categorical_features=categorical_features,verbose=True,mode='regression')




#Pick an instance to explain
j = np.random.randint(0, X_test.shape[0])
# optional, set j manually
j = 0
# initialize js
shap.initjs()



#XGBoost::
#SHAP
shap.force_plot(explainerXGB.expected_value, shap_values_XGB_test[j], X_test.iloc[[j]])

#LIME
#Out-of-the-box LIME cannot handle the requirement of XGBoost to use xgb.DMatrix() on the input data

# the predict function input doesn't jive wtih LIME
xgb_model.predict(xgb.DMatrix(X_test.iloc[[j]]))
# this will throw an error
"""
expXGB = explainer.explain_instance(X_test.values[j], xgb_model.predict, num_features=5)
expXGB.show_in_notebook(show_table=True)
"""



#Scikit-learn GBT::
#SHAP
shap.force_plot(explainerSKGBT.expected_value, shap_values_SKGBT_test[j], X_test.iloc[[j]])

#LIME
#------------------------------
'''expSKGBT = explainer.explain_instance(X_test.values[j], sk_xgb.predict, num_features=5)
expSKGBT.show_in_notebook(show_table=True)'''



#Random Forest::
#SHAP
shap.force_plot(explainerRF.expected_value, shap_values_RF_test[j], X_test.iloc[[j]])

#LIME
#------------------------------
'''exp = explainer.explain_instance(X_test.values[j], rf.predict, num_features=5)
exp.show_in_notebook(show_table=True)'''



#KNN::
#SHAP
shap.force_plot(explainerKNN.expected_value, shap_values_KNN_test[j], X_test.iloc[[j]])
#------------------------------
#LIME
'''exp = explainer.explain_instance(X_test.values[j], knn.predict, num_features=5)
exp.show_in_notebook(show_table=True)'''





#Explaining the Global Model

#Importance plot via SHAP values
#------------------------------
#shap.summary_plot(shap_values_XGB_train, X_train, plot_type="bar")

#Similar to variable importance, this shows the SHAP values for every instance from the training dataset
#------------------------------
#shap.summary_plot(shap_values_XGB_train, X_train)

#Variable Influence or Dependency Plots
#Default SHAP dependency plot

#------------------------------
#shp_plt = shap.dependence_plot("LSTAT", shap_values_XGB_train, X_train)
#The following modifies the default graph a bit to (1) highlight the jth instance with a black dot and (2) pick the color by variable ourselves

# inputs = column of interest as string, column for coloring as string, df of our data, SHAP df, 
#          x postion of the black dot, y position of the black dot

def dep_plt(col, color_by, base_actual_df, base_shap_df, overlay_x, overlay_y):
    cmap=sns.diverging_palette(260, 10, sep=1, as_cmap=True) #seaborn pallete
    f, ax = plt.subplots()
    points = ax.scatter(base_actual_df[col], base_shap_df[col], c=base_actual_df[color_by], s=20, cmap=cmap)
    f.colorbar(points).set_label(color_by)
    ax.scatter(overlay_x, overlay_y, color='black', s=50)
    plt.xlabel(col)
    plt.ylabel("SHAP value for " + col)
    plt.show()
# get list of model inputs in order of SHAP importance
imp_cols = df_shap_XGB_train.abs().mean().sort_values(ascending=False).index.tolist()

# loop through this list to show top 3 dependency plots
for i in range(0, len(imp_cols)):
    #plot the top var and color by the 2nd var
    if i == 0 : 
        dep_plt(imp_cols[i], 
                imp_cols[i+1], 
                X_train, 
                df_shap_XGB_train,
                X_test.iloc[j,:][imp_cols[i]], 
                df_shap_XGB_test.iloc[j,:][imp_cols[i]])
    #plot the 2nd and 3rd vars and color by the top var
    if (i > 0) and (i < 3) : 
        dep_plt(imp_cols[i], 
                imp_cols[0], 
                X_train, 
                df_shap_XGB_train,
                X_test.iloc[j,:][imp_cols[i]], 
                df_shap_XGB_test.iloc[j,:][imp_cols[i]])
