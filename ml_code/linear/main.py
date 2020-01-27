import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
import sklearn
import shap
%matplotlib inline

data = pd.read_csv("kc_house_data.csv")

#data.head()
#data.describe()

#data['bedrooms'].value_counts().plot(kind='bar')
#plt.title('number of Bedroom')
#plt.xlabel('Bedrooms')
#plt.ylabel('Count')
#sns.despine

#plt.figure(figsize=(10,10))
#sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
#plt.ylabel('Longitude', fontsize=12)
#plt.xlabel('Latitude', fontsize=12)
#plt.show()
#plt1 = plt()
#sns.despine

#plt.scatter(data.price,data.sqft_living)
#plt.title("Price vs Square Feet")

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)
reg.fit(x_train,y_train)
reg.score(x_test,y_test)


explainer = shap.LinearExplainer(reg,x_train, feature_dependence="independent")
shap_values = explainer.shap_values(x_test)
X_test_array = np.asarray(x_test)
#X_test_array = x_test.toarray()# we need to pass a dense version for the plotting functions
names=[]
for i in x_test:
    names.append(i)  

#plotting
shap.summary_plot(shap_values,X_test_array,feature_names=names)
shap.initjs()
ind = 0
print(shap_values[ind,:])
print(X_test_array[ind,:])
print(explainer.expected_value)
'''shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test_array[ind,:],
    feature_names=names)'''
