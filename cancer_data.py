#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("/home/ashraf/projects/privacy/data.csv")
df.head()


# In[3]:


len(df.index), len(df.columns)


# In[4]:


df.shape


# In[5]:


clear()


# In[6]:


df.head()


# In[7]:


import pandas as pd
df = pd.read_csv("/home/ashraf/projects/privacy/updated.csv")
df.head()


# In[8]:


df.shape()


# In[9]:


df.shape


# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df.info()


# In[13]:


df.isna()


# In[14]:


print(df['target_names'])


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[16]:


from sklearn.datasets import load_breast_cancer


# In[17]:


cancer = load_breast_cancer()


# In[18]:


cancer


# In[19]:


cancer.keys()


# In[20]:


print(cancer['DESCR'])


# In[21]:


print(cancer['target'])


# In[22]:


print(cancer['target_names'])


# In[23]:


print(cancer['feature_names'])


# In[24]:


cancer['data'].shape


# In[25]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'],['target']))


# In[26]:


df_cancer.head()


# In[27]:


df_cancer.tail()


# In[28]:


sns.pairplot(df_cancer , vars =['mean radius','mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness' ,'mean concavity',
 'mean concave points', 'mean symmetry', 'mean fractal dimension',
 'radius error', 'texture error', 'perimeter error', 'area error',
 'smoothness error', 'compactness error', 'concavity error',
 'concave points error', 'symmetry error', 'fractal dimension error',
 'worst radius', 'worst texture', 'worst perimeter', 'worst area',
 'worst smoothness', 'worst compactness', 'worst concavity',
 'worst concave points', 'worst symmetry', 'worst fractal dimension'])  


# In[29]:


sns.pairplot(df_cancer ,hue ='target', vars =['mean radius','mean texture', 'mean perimeter', 'mean area',
 'mean smoothness', 'mean compactness' ,'mean concavity']) 


# In[30]:


sns.countplot(df_cancer['target'])


# In[31]:


sns.scatterplot(x='mean area',y='mean smoothness',hue='target',data =df_cancer)


# In[34]:


plt.figure(figsize =(20,10))
sns.heatmap(df_cancer.corr(), annot =True)


# In[35]:


x = df_cancer.drop(['target'],axis =1)


# In[36]:


x


# In[37]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[38]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[39]:


data.columns


# In[40]:


cancer['data'].columns


# In[41]:


df.columns


# In[42]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[44]:


X = df[prediction_feature]
X


# In[45]:


y = df.diagnosis
y


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=15)


# In[47]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=15)


# In[49]:


print(X_train)


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[51]:


print(X_train)


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)


# In[59]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[60]:


def model_building(model, X_train, X_test, y_train, y_test):
    """
    
    Model Fitting, Prediction And Other stuff
    return ('score', 'accuracy_score', 'predictions' )
    """
    
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    
    return (score, accuracy, predictions)    


# In[61]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[62]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[63]:


def cm_metrix_graph(cm):
    
    sns.heatmap(cm,annot=True,fmt="d")
    plt.show()


# In[64]:


df_prediction = []
confusion_matrixs = []
df_prediction_cols = [ 'model_name', 'score', 'accuracy_score' , "accuracy_percentage"]

for name, model in zip(list(models_list.keys()), list(models_list.values())):
    
    (score, accuracy, predictions) = model_building(model, X_train, X_test, y_train, y_test )
    
    print("\n\nClassification Report of '"+ str(name), "'\n")
    
    print(classification_report(y_test, predictions))

    df_prediction.append([name, score, accuracy, "{0:.2%}".format(accuracy)])
    
    # For Showing Metrics
    confusion_matrixs.append(confusion_matrix(y_test, predictions))
    
        
df_pred = pd.DataFrame(df_prediction, columns=df_prediction_cols)


# In[65]:


print(len(confusion_matrixs))


# In[68]:


df_pred


# In[69]:


data = pd.read_csv("/home/ashraf/projects/privacy/synthetic data.csv")


# In[70]:


len(data.index), len(data.columns)


# In[71]:


data.head()


# In[72]:


data.info()


# In[73]:


data.isna()


# In[74]:


data.isna().any()


# In[75]:


data.isna().sum() 


# In[76]:


data = data.dropna(axis='columns')


# In[77]:


data.describe(include="O")


# In[78]:


data.diagnosis.value_counts()


# In[79]:


data.head(2)


# In[80]:


diagnosis_unique = data.diagnosis.unique()


# In[81]:


diagnosis_unique


# In[83]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[84]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[85]:


# plt.figure(figsize=(7,12))
px.histogram(data, x='diagnosis')
# plt.show()


# In[86]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[87]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[88]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[89]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[90]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[91]:


data.columns


# In[92]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[93]:


X = data[prediction_feature]
X


# In[94]:


y = data.diagnosis
y


# In[95]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)
# print(X_test)


# In[96]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[97]:


def model_building(model, X_train, X_test, y_train, y_test):
    """
    
    Model Fitting, Prediction And Other stuff
    return ('score', 'accuracy_score', 'predictions' )
    """
    
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    
    return (score, accuracy, predictions)   


# In[98]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[99]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[100]:


def cm_metrix_graph(cm):
    
    sns.heatmap(cm,annot=True,fmt="d")
    plt.show()
        


# In[101]:


df_prediction = []
confusion_matrixs = []
df_prediction_cols = [ 'model_name', 'score', 'accuracy_score' , "accuracy_percentage"]

for name, model in zip(list(models_list.keys()), list(models_list.values())):
    
    (score, accuracy, predictions) = model_building(model, X_train, X_test, y_train, y_test )
    
    print("\n\nClassification Report of '"+ str(name), "'\n")
    
    print(classification_report(y_test, predictions))

    df_prediction.append([name, score, accuracy, "{0:.2%}".format(accuracy)])
    
    # For Showing Metrics
    confusion_matrixs.append(confusion_matrix(y_test, predictions))
    
        
df_pred = pd.DataFrame(df_prediction, columns=df_prediction_cols)


# In[102]:


df_pred


# In[ ]:




