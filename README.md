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


**ML References**:
1. [Siraj Raval YouTube playlist](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A/playlists)
2. [Bhavesh Bhatt YouTube Playlist](https://www.youtube.com/channel/UC8ofcOdHNINiPrBA9D59Vaw/playlists)
3. [ML YouTube Playlist](https://www.youtube.com/channel/UCyHta2dyCTkf29AB67AYn7A/playlists)
4. [Udemy: Machine Learning A-Z by Kirill Eremenko](https://www.udemy.com/machinelearning/learn/lecture/6087180#overview)
5. [Udemy: ML Feature Selection by Soledad Galli](https://www.udemy.com/feature-selection-for-machine-learning/learn/lecture/9341678#overview)

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
4. Explore/Implement SHAP or LIME for below algorithms, as per your choice of use-cases. (Can distribute algorithm specific work among yourself);
   - Linear regression.
   - Logistic regression.
   - Decision trees/Random Forest.
   - SVM
   - Object detection/classification.
5. Check if existing model can be customised using SHAP/LIME.
6. Explore, if you could find some other methods for Explainable.AI other than SHAP/LIME.
7. Create issue tickets in case of any issues/difficulties.

