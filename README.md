# PSL_VNIT_ML_PROJ
----

**Contributor's**:
1. [Sathishreddy Gathpa](sathishreddygathpa@gmail.com)
2. [Pranay Bunari](pranaybunari@gmail.com)
3. [Eshwar Koride](eshwarkoride7799@gmail.com)
4. [Gangisetty Sai Karthik](saikarthik3099@gmail.com)

**Mentor**:
1. [Saurabh Jain](saurabh.vnit@gmail.com) and [Hemant Kshirsagar](hemantkshirsagar8@gmail.com)

----

Problem Statement|Summary|Description|Status
---|---|---|---|
Explainable AI - Framework to inspect AI Model to explain output|Framework to support various types of ML models (including Deep Learning models) to inspect features that have contributed to the algorithm output. Use this to explain algorithm output|Create a framework which explains, details out the reason for a model output. This explanation will be in terms of various features used for model training and/or annotations done. Various steps will need to be considered – 1. Figure out various ML algorithms like Linear Regression, Logistic regression, Classification, Image classification using Deep Learning techniques, NLP tasks as NER etc. 2. For each of these ways need to figure out the best way to intuitively explain the model output to end consumer of ML models assuming end user is non-technical. 3. Create a tool/framework/POC, input to which will be ML model and output will be an UI explaining the output along with names, weights, significance of each feature being considered for this decision.|Taken for work
NLP Analytics - Entity Knowledge Graph|Creating NLP algorithm to extract Knowledge graph on documents|Create a Tool that can parse documents that are related to a certain domain and extract the primary entities out of the document along with relationships between the entities, creating a Knowledge graph of the domain. For example if the tool is given a set of documents that are related to the Health Care domain it should be able to identify Patients, Doctors, Disease, Prescription as the entities and the relationship between them.|Not taken yet. 


----

**References**:
1. [A Brief History of Machine Learning Models Explainability](https://medium.com/@Zelros/a-brief-history-of-machine-learning-models-explainability-f1c3301be9dc)
2. [SHAP Git-Hub](https://github.com/slundberg/shap)
3. [SHAP paper](https://arxiv.org/abs/1705.07874)
4. [LIME](https://arxiv.org/abs/1602.04938)
5. [MLFlow](https://mlflow.org/docs/latest/quickstart.html)
6. [Why MLFlow?](https://www.youtube.com/watch?v=QJW_kkRWAUs)

----

**System pre-requisite**:
1. Windows 10/Ubuntu 18.04 (As per you prefer.)
2. 64 bit >16GB RAM
3. Latest Anaconda
4. Python 3.6
5. PyCharm
6. Jupyter Notebook (comes with Anaconda)

----

As per 'Sathishreddy Gathpa' on '11th Sep 19' :
>So we have finally came to a conclusion that  we take up a real life unsolved problem and model it from scratch and explaining it through AI  as our project. 
>We find this interesting and also we get a chance to dig deeper into the same instead of explaining the problems which have been already solved.


----

**Lets implement code**:

1. Setup environment in your machine Or can use [google colab](https://colab.research.google.com/) **(Colab is recommended due to GPU support and its free.)** 
2. Practice, implement and understand SHAP, as mentioned [here](https://github.com/slundberg/shap)
3. In above link, SHAP implementation and examples are provided for xgboost, tensorflow object classification and  VGG16 model.
4. Explore/Implement SHAP or LIME for below algorithms, as per your choice of use-cases. Distribute algorithm specific workamong yourself;

  4.1. Linear regression.
  4.2. Logistic regression.
  4.3. Decision trees/Random Forest.
  4.4. SVM
  4.5. Object detection/classification.
  
5. Check if existing model can be customised using SHAP/LIME.
6. Explore, if you could find some other methods for Explainable.AI other than SHAP/LIME.
7. Create issue ticket in case of any issues/difficulties.

