#!/usr/bin/env python
# coding: utf-8

# In[1]:


data = pd.read_csv("/home/ashraf/projects/privacy/updated.csv")


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/updated.csv")


# In[ ]:


len(data.index), len(data.columns)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[2]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data = data.dropna(axis='columns')


# In[ ]:


data.describe(include="O")


# In[ ]:


data.diagnosis.value_counts()


# In[ ]:


diagnosis_unique = data.diagnosis.unique()


# In[ ]:


diagnosis_unique


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[ ]:


import plotly.express as px
import plotly.graph_objects as go

# plt.figure(figsize=(7,12))
px.histogram(data, x='diagnosis')
# plt.show()


# In[ ]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[ ]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[ ]:


data.head(2)


# In[ ]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[ ]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[ ]:


plt.figure(figsize=(10, 8))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[ ]:


plt.figure(figsize=(15, 10))


fig = px.imshow(data[cols].corr());
fig.show()


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[ ]:


data.columns


# In[ ]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[ ]:


X = data[prediction_feature]
X

# print(X.shape)
# print(X.values)


# In[ ]:


y = data.diagnosis
y

# print(y.values)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)
# print(X_test)


# In[3]:


# Scale the data to keep all the values in the same magnitude of 0 -1 

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


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


# In[ ]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[ ]:


print(list(models_list.keys()))
print(list(models_list.values()))

# print(zip(list(models_list.keys()), list(models_list.values())))


# In[ ]:


# Let's Define the function for confision metric Graphs

def cm_metrix_graph(cm):
    
    sns.heatmap(cm,annot=True,fmt="d")
    plt.show()
        


# In[ ]:


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


# In[ ]:


print(len(confusion_matrixs))


# In[ ]:


df_pred


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/synthetic data.csv")


# In[ ]:


len(data.index), len(data.columns)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data = data.dropna(axis='columns')


# In[ ]:


data.describe(include="O")


# In[ ]:


data.diagnosis.value_counts()


# In[ ]:


diagnosis_unique = data.diagnosis.unique()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[ ]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[ ]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


data.head(2)


# In[ ]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[ ]:


data.head(2)


# In[ ]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[ ]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[ ]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[4]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[5]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[6]:


data.columns


# In[ ]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[ ]:


X = data[prediction_feature]
X


# In[ ]:


y = data.diagnosis
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)


# In[ ]:


# Scale the data to keep all the values in the same magnitude of 0 -1 

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


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


# In[ ]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[ ]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[ ]:


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


# In[ ]:


df_pred


# In[ ]:


df_pred


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/Breast_cancer_data.csv")


# In[ ]:


len(data.index), len(data.columns)


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data = data.dropna(axis='columns')


# In[ ]:


data.describe(include="O")


# In[ ]:


data.diagnosis.value_counts()


# In[ ]:


data.head(2)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[ ]:


cols = ["diagnosis", "mean_radius", "mean_texture", "mean_perimeter", "mean_area","mean_smoothness"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/dataR2.csv")


# In[ ]:


len(data.index), len(data.columns)


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data.Classification.value_counts()


# In[ ]:


data.head(2)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist( data.Classification)
# plt.legend()
plt.title("Counts of Classification")
plt.xlabel("Classification")


plt.subplot(1, 2, 2)

sns.countplot('Classification', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show()


# In[ ]:


cols = ["Age", "BMI", "Glucose", "Insulin", "HOMA","Leptin","Adiponectin","Resistin","MCP.1","Classification"]

sns.pairplot(data[cols], hue="Classification")
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


data.head(2)


# In[ ]:


labelencoder_Y = LabelEncoder()
data.Classification = labelencoder_Y.fit_transform(data.Classification)


# In[ ]:


data.head(2)


# In[ ]:


print(data.Classification.value_counts())
print("\n", data.Classification.value_counts().sum())


# In[ ]:


cols = ['Classification', 'Age', 'BMI', 'Glucose',
       'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']
print(len(cols))
data[cols].corr()


# In[ ]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[ ]:


prediction_feature = ['Age', 'BMI', 'Glucose',
       'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']

targeted_feature = 'Classification'

len(prediction_feature)


# In[ ]:


X = data[prediction_feature]
X


# In[ ]:


y = data.Classification
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)


# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


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


# In[ ]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[ ]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[ ]:


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


# In[ ]:


df_pred


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/third.csv")


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data.describe(include="O")


# In[ ]:


data.Type.of.Death.value_counts()


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/second_dataR2.csv")


# In[ ]:


len(data.index), len(data.columns)


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data.Classification.value_counts()


# In[ ]:


data.head(2)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist( data.Classification)
# plt.legend()
plt.title("Counts of Classification")
plt.xlabel("Classification")


plt.subplot(1, 2, 2)

sns.countplot('Classification', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show()


# In[ ]:


cols = ["Age", "BMI", "Glucose", "Insulin", "HOMA","Leptin","Adiponectin","Resistin","MCP.1","Classification"]

sns.pairplot(data[cols], hue="Classification")
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


data.head(2)


# In[ ]:


labelencoder_Y = LabelEncoder()
data.Classification = labelencoder_Y.fit_transform(data.Classification)


# In[ ]:


data.head(2)


# In[ ]:


print(data.Classification.value_counts())
print("\n", data.Classification.value_counts().sum())


# In[ ]:


cols = ['Classification', 'Age', 'BMI', 'Glucose',
       'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']
print(len(cols))
data[cols].corr()


# In[ ]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[ ]:


prediction_feature = ['Age', 'BMI', 'Glucose',
       'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']

targeted_feature = 'Classification'

len(prediction_feature)


# In[ ]:


X = data[prediction_feature]
X


# In[ ]:


y = data.Classification
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)


# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


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


# In[ ]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[ ]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[ ]:


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


# In[ ]:


df_pred


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/1.csv")


# In[ ]:


len(data.index), len(data.columns)


# In[ ]:


data.head()


# In[ ]:


data
.
head
(
)


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[7]:


data.isna().sum() 


# In[ ]:


data = data.dropna(axis='columns')


# In[ ]:


data.describe(include="O")


# In[ ]:


data.diagnosis.value_counts()


# In[ ]:


diagnosis_unique = data.diagnosis.unique()


# In[ ]:


diagnosis_unique


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[ ]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[ ]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[ ]:


data.head(2)


# In[ ]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[ ]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[ ]:


plt.figure(figsize=(10, 8))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[ ]:


data.info()


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/1.csv")


# In[ ]:


len(data.index), len(data.columns)


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data.describe(include="O")


# In[ ]:


data.diagnosis.value_counts()


# In[ ]:


data.head(2)


# In[ ]:


diagnosis_unique = data.diagnosis.unique()


# In[ ]:


diagnosis_unique


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[ ]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[ ]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


data.head(2)


# In[ ]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[ ]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[ ]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[ ]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[8]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[9]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[10]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[11]:


data.columns


# In[ ]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[ ]:


X = data[prediction_feature]
X


# In[ ]:


y = data.diagnosis
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)


# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


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


# In[ ]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[ ]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[ ]:


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


# In[ ]:


def cm_metrix_graph(cm):
    
    sns.heatmap(cm,annot=True,fmt="d")
    plt.show()


# In[ ]:


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


# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


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


# In[ ]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[ ]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[ ]:


# Let's Define the function for confision metric Graphs

def cm_metrix_graph(cm):
    
    sns.heatmap(cm,annot=True,fmt="d")
    plt.show()
        


# In[ ]:


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


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data = data.dropna(axis='columns')


# In[ ]:


data.describe(include="O")


# In[ ]:


data.head(2)


# In[ ]:


diagnosis_unique = data.diagnosis.unique()


# In[ ]:


diagnosis_unique


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[ ]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[ ]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


data.head(2)


# In[ ]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[ ]:


data.head(2)


# In[ ]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[12]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[ ]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
     'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[ ]:


plt.figure(figsize=(10, 8))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[ ]:


data.columns


# In[ ]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[ ]:


X = data[prediction_feature]
X


# In[ ]:


y = data.diagnosis
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)
# print(X_test)


# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


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


# In[ ]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[ ]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[ ]:


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


# In[ ]:


df_pred


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/2.csv")


# In[ ]:


len(data.index), len(data.columns)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data.Classification.value_counts()


# In[ ]:


data.head(2)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.Classification)
# plt.legend()
plt.title("Counts of Classification")
plt.xlabel("Classification")


plt.subplot(1, 2, 2)

sns.countplot('Classification', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show()


# In[ ]:


cols = ["Age", "BMI", "Glucose", "Insulin", "HOMA","Leptin","Adiponectin","Resistin","MCP.1","Classification"]

sns.pairplot(data[cols], hue="Classification")
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


data.head(2)


# In[ ]:


labelencoder_Y = LabelEncoder()
data.Classification = labelencoder_Y.fit_transform(data.Classification)


# In[ ]:


data.head(2)


# In[ ]:


print(data.Classification.value_counts())
print("\n", data.Classification.value_counts().sum())


# In[ ]:


cols = ['Classification', 'Age', 'BMI', 'Glucose',
       'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']
print(len(cols))
data[cols].corr()


# In[13]:


plt.figure(figsize=(10, 8))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[ ]:


prediction_feature = ['Age', 'BMI', 'Glucose',
       'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']

targeted_feature = 'Classification'

len(prediction_feature)


# In[ ]:


X = data[prediction_feature]
X


# In[ ]:


y = data.Classification
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)


# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


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


# In[ ]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[ ]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[ ]:


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


# In[ ]:


df_pred


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/third.csv")


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data.describe(include="O")


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/third.csv")


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data = data.dropna(axis='columns')


# In[ ]:


data.diagnosis.value_counts()


# In[ ]:


data.head(2)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.cancer)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("cancer")


plt.subplot(1, 2, 2)

sns.countplot('cancer', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>


# In[ ]:


cols = ["Age", "Number_of_sexual_partners", "First_sexual_intercourse", "Number_of_pregnancies", "Smokes", "Smokes_year","Smokes_packets_per_year"]

sns.pairplot(data[cols], hue="cancer")
plt.show()


# In[14]:


cols = ["Age", "Number_of_sexual_partners", "First_sexual_intercourse", "Number_of_pregnancies", "Smokes", "Smokes_year","Smokes_packets_per_year"]

sns.pairplot(data[cols], hue="cancer")
plt.show()


# In[ ]:


cols = ["Age", "Number_of_sexual_partners", "First_sexual_intercourse", "Number_of_pregnancies", "Smokes", "Smokes_year"]

sns.pairplot(data[cols], hue="cancer")
plt.show()


# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.cancer)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("cancer")


plt.subplot(1, 2, 2)

sns.countplot('cancer', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


data.head(2)


# In[ ]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.cancer)


# In[ ]:


labelencoder_Y = LabelEncoder()
data.cancer = labelencoder_Y.fit_transform(data.cancer)


# In[ ]:


data.head(2)


# In[ ]:


print(data.cancer.value_counts())
print("\n", data.cancer.value_counts().sum())


# In[ ]:


plt.figure(figsize=(10, 8))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[ ]:


data.columns


# In[ ]:


X = data[prediction_feature]
X


# In[ ]:


prediction_feature = [ "Age",  'Number_of_sexual_partners', 'First_sexual_intercourse', 'Number_of_pregnancies', 'Hormonal_Contraceptives', 'Hormonal_Contraceptives_year']

targeted_feature = 'cancer'

len(prediction_feature)


# In[ ]:


X = data[prediction_feature]
X


# In[ ]:


y = data.diagnosis
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)


# In[ ]:



sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/third.csv")


# In[ ]:


len(data.index), len(data.columns)


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


data.cancer.value_counts()


# In[ ]:


diagnosis_unique = data.cancer.unique()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.cancer)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Cancer")


plt.subplot(1, 2, 2)

sns.countplot('cancer', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[ ]:


cols = ["Age", "Number_of_sexual_partners", "Number_of_pregnancies", "Smokes", "Smokes_year","Smokes_packets_per_year","cancer"]

sns.pairplot(data[cols], hue="cancer")
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


labelencoder_Y = LabelEncoder()
data.cancer = labelencoder_Y.fit_transform(data.cancer)


# In[ ]:


print(data.cancer.value_counts())
print("\n", data.cancer.value_counts().sum())


# In[ ]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[15]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[16]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[17]:


prediction_feature = [ "Age",  'Number_of_sexual_partners', 'First_sexual_intercourse', 'Number_of_pregnancies', 'Smokes', 'Smokes_year']

targeted_feature = 'cancer'

len(prediction_feature)


# In[18]:


X = data[prediction_feature]
X


# In[ ]:


y = data.cancer
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)


# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


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


# In[ ]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[ ]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[ ]:


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


# In[ ]:


df_pred


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/3.csv")


# In[ ]:


data.info()


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist( data.cancer)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("cancer")


plt.subplot(1, 2, 2)

sns.countplot('cancer', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


labelencoder_Y = LabelEncoder()
data.cancer = labelencoder_Y.fit_transform(data.cancer)


# In[ ]:


print(data.cancer.value_counts())
print("\n", data.cancer.value_counts().sum())


# In[ ]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScale


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[ ]:


prediction_feature = [ "Age",  'Number_of_sexual_partners', 'First_sexual_intercourse', 'Number_of_pregnancies', 'Smokes', 'Smokes_year']

targeted_feature = 'cancer'

len(prediction_feature)


# In[ ]:


X = data[prediction_feature]
X


# In[ ]:


y = data.diagnosis
y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)
# print(X_test)


# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


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


# In[ ]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[ ]:


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


# In[ ]:


df_pred


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv


# In[ ]:


data = pd.read_csv("/home/ashraf/projects/privacy/50000.csv")


# In[ ]:


data.isna()


# In[ ]:


data.isna().any()


# In[ ]:


data.isna().sum() 


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.cancer)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("cancer")


plt.subplot(1, 2, 2)

sns.countplot('cancer', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[ ]:


import math
import time
import numpy as np
import matplotlib.pyplot as plt

def doSimulation(K, T, S, r, sigma, delta, N, M):
	t1 = time.perf_counter_ns()

	dt = T/N
	nudt = (r - delta - ((sigma**2)/2))*dt
	sigsdt = sigma*(dt**0.5)
	lnS = np.log(S)
	sum_CT = 0
	sum_CT2 = 0

	for j in range(1, M+1):
		lnSt = lnS
		for i in range(1, N+1):
			e = 0.4
			lnSt = lnSt + nudt + sigsdt*e
		ST = math.exp(lnSt)
		CT = max(0, ST - K)
		sum_CT = sum_CT + CT
		sum_CT2 = sum_CT2 + (CT**2)

	option_value = (sum_CT/M)*math.exp(-1*r*T)
	#SD = math.sqrt((sum_CT2 - (sum_CT**2)/M)*(math.exp(-2*r*t)/(M-1)))
	#SE = SD/math.sqrt(M)

	t2 = time.perf_counter_ns()

	return option_value, t2 - t1


if __name__ == '__main__':
	S = 100
	T = 1
	r = 0.12
	sigma = 0.20
	delta = 0.03

	K = 120
	M = 100
	print("Varying the number of timesteps (N) with a fixed number of simulations:", M)
	N_list = [2, 5, 10, 15, 20, 25, 30]
	value_list = []
	for N in N_list:
		value, ex_time = doSimulation(K, T, S, r, sigma, delta, N, M)
		value_list.append(value)
		print("Execution time for timestep = {0}: {1} nano seconds".format(N, ex_time))

	plt.plot(N_list, value_list)
	plt.title("Asset price values for various timesteps")
	plt.xlabel("Timesteps")
	plt.ylabel("Asset price values")
	plt.show()
	plt.savefig('problem_1_a.png')

	print("\n")

	N = 20
	print("Varying the number of simulations (M) with a fixed number of timesteps:", N)
	M_list = [10, 20, 30, 50, 75, 100, 150, 200]
	time_list = []
	for M in M_list:
		value, ex_time = doSimulation(K, T, S, r, sigma, delta, N, M)
		time_list.append(ex_time)
		print("Asset price for M = {0}: {1}".format(M, value))

	plt.plot(M_list, time_list)
	plt.title("Execution time for various number of simulations (M)")
	plt.xlabel("Number of simulations (M)")
	plt.ylabel("Execution time in nano seconds")
	plt.show()
	plt.savefig('problem_1_b.png')






# In[ ]:


import math
import time
import numpy as np
import matplotlib.pyplot as plt

def getEuropeanCall(K, T, S, r, sigma, dx, v, N, Nj):
	t1 = time.perf_counter_ns()

	dt = T/N
	edx = math.exp(dx)
	pu = (dt*(((sigma/dx)**2) + (v/dx)))/2
	pm = 1 - dt*((sigma/dx)**2) - r*dt
	pd = (dt*(((sigma/dx)**2) - (v/dx)))/2

	St = [0]*(Nj + 1)
	St_minus = [0]*(Nj + 1)
	St_minus[Nj] = S*math.exp(-1*Nj*dx)
	for j in range(-1*Nj + 1, Nj + 1):
		if j-1 < 0:
			value = St_minus[-1*(j-1)]*edx
		else:
			value = St[j-1]*edx
		if j < 0:
			St_minus[-1*j] = value
		else:
			St[j] = value

	C = [[0]*(Nj + 1) for i in range(N + 1)]
	C_minus = [[0]*(Nj + 1) for i in range(N + 1)]
	for j in range(-1*Nj, Nj + 1):
		if j < 0:
			C_minus[N][-1*j] = max(0, St_minus[-1*j] - K)
		else:
			C[N][j] = max(0, St[j] - K)

	for i in range(N-1, -1, -1):
		for j in range(-1*Nj + 1, Nj):
			if j + 1 < 0:
				firstPart = pu*C_minus[i + 1][-1*(j + 1)]
			else:
				firstPart = pu*C[i + 1][j + 1]
			if j < 0:
				secondPart = pm*C_minus[i + 1][-1*j]
			else:
				secondPart = pm*C[i + 1][j]
			if j - 1 < 0:
				thirdPart = pd*C_minus[i + 1][-1*(j - 1)]
			else:
				thirdPart = pd*C[i + 1][j - 1]
			if j < 0:
				C_minus[i][-1*j] = firstPart + secondPart + thirdPart
			else:
				C[i][j] = firstPart + secondPart + thirdPart

		C_minus[i][-1*Nj] = C_minus[i][-1*Nj + 1]
		C[i][Nj] = C[i][Nj - 1] + (St[Nj] - St[Nj - 1])


	t2 = time.perf_counter_ns()

	return C[0][0], t2 - t1


if __name__ == '__main__':
	S = 100
	T = 1
	r = 0.12
	sigma = 0.20
	dx = 0.02
	v = 0.08

	K = 120
	Nj = 25
	print("Varying the number of timesteps (N) with a fixed Strike Price:", S)
	N_list = [5, 10, 15, 20, 25, 30, 35, 40, 45]
	value_list = []
	for N in N_list:
		value, ex_time = getEuropeanCall(K, T, S, r, sigma, dx, v, N, Nj)
		value_list.append(value)
		print("Execution time for timestep = {0}: {1} nano seconds".format(N, ex_time))

	plt.plot(N_list, value_list)
	plt.title("European call values for various timesteps")
	plt.xlabel("Timesteps")
	plt.ylabel("European call values")
	plt.show()
	plt.savefig('problem_2_a.png')

	print("\n")

	N = 20
	print("Varying the Strike Price (S) with a fixed number of timesteps:", N)
	S_list = [70, 80, 90, 100, 110, 120, 130]
	value_list = []
	for S in S_list:
		value, ex_time = getEuropeanCall(K, T, S, r, sigma, dx, v, N, Nj)
		value_list.append(value)
		print("Execution time for S = {0}: {1} nano seconds".format(S, ex_time))

	plt.plot(S_list, value_list)
	plt.title("European call values for various Strike Price (S)")
	plt.xlabel("Value of Strike Price (S)")
	plt.ylabel("European call value")
	plt.show()
	plt.savefig('problem_2_b.png')







# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr

# import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)
# Monte Carlo Method
mc_sims = 400 # number of simulations
T = 100 #timeframe in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))#uncorrelated RV's
    L = np.linalg.cholesky(covMatrix) #Cholesky decomposition to Lower Triangular Matrix
    dailyReturns = meanM + np.inner(L, Z) #Correlated daily returns for individual stocks
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()


# In[ ]:


import numpy as np
from scipy.stats import norm

N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)
K = 100
r = 0.1
T = 1
sigma = 0.3

S = np.arange(60,140,0.1)

calls = [BS_CALL(s, K, T, r, sigma) for s in S]
puts = [BS_PUT(s, K, T, r, sigma) for s in S]
plt.plot(STs, calls, label='Call Value')
plt.plot(STs, puts, label='Put Value')
plt.xlabel('$S_0$')
plt.ylabel(' Value')
plt.legend()


# In[ ]:


import math
import time
import numpy as np
import matplotlib.pyplot as plt

def doSimulation(K, T, S, r, sigma, delta, N, M):
	t1 = time.perf_counter_ns()

	dt = T/N
	nudt = (r - delta - ((sigma**2)/2))*dt
	sigsdt = sigma*(dt**0.5)
	lnS = np.log(S)
	sum_CT = 0
	sum_CT2 = 0

	for j in range(1, M+1):
		lnSt = lnS
		for i in range(1, N+1):
			e = 0.4
			lnSt = lnSt + nudt + sigsdt*e
		ST = math.exp(lnSt)
		CT = max(0, ST - K)
		sum_CT = sum_CT + CT
		sum_CT2 = sum_CT2 + (CT**2)

	option_value = (sum_CT/M)*math.exp(-1*r*T)
	#SD = math.sqrt((sum_CT2 - (sum_CT**2)/M)*(math.exp(-2*r*t)/(M-1)))
	#SE = SD/math.sqrt(M)

	t2 = time.perf_counter_ns()

	return option_value, t2 - t1


if __name__ == '__main__':
	S = 100
	T = 1
	r = 0.12
	sigma = 0.20
	delta = 0.03

	K = 120
	M = 100
	print("Varying the number of timesteps (N) with a fixed number of simulations:", M)
	N_list = [2, 5, 10, 15, 20, 25, 30]
	value_list = []
	for N in N_list:
		value, ex_time = doSimulation(K, T, S, r, sigma, delta, N, M)
		value_list.append(value)
		print("Execution time for timestep = {0}: {1} nano seconds".format(N, ex_time))

	plt.plot(N_list, value_list)
	plt.title("Asset price values for various timesteps")
	plt.xlabel("Timesteps")
	plt.ylabel("Asset price values")
	plt.show()
	plt.savefig('problem_1_a.png')

	print("\n")

	N = 20
	print("Varying the number of simulations (M) with a fixed number of timesteps:", N)
	M_list = [10, 20, 30, 50, 75, 100, 150, 200]
	time_list = []
	for M in M_list:
		value, ex_time = doSimulation(K, T, S, r, sigma, delta, N, M)
		time_list.append(ex_time)
		print("Asset price for M = {0}: {1}".format(M, value))

	plt.plot(M_list, time_list)
	plt.title("Execution time for various number of simulations (M)")
	plt.xlabel("Number of simulations (M)")
	plt.ylabel("Execution time in nano seconds")
	plt.show()
	plt.savefig('problem_1_b.png')






# In[22]:


import numpy as np
import matplotlib.pyplot as plt


def geo_paths(S, T, r, q, sigma, steps, N):
    """
    Inputs
    #S = Current stock Price
    #K = Strike Price
    #T = Time to maturity 1 year = 1, 1 months = 1/12
    #r = risk free interest rate
    #q = dividend yield
    # sigma = volatility 
    
    Output
    # [steps,N] Matrix of asset paths 
    """
    dt = T/steps
    #S_{T} = ln(S_{0})+\int_{0}^T(\mu-\frac{\sigma^2}{2})dt+\int_{0}^T \sigma dW(t)
    ST = np.log(S) +  np.cumsum(((r - q - sigma**2/2)*dt +                              sigma*np.sqrt(dt) *                               np.random.normal(size=(steps,N))),axis=0)
    
    return np.exp(ST)

S = 100 #stock price S_{0}
K = 110 # strike
T = 1/2 # time to maturity
r = 0.05 # risk free risk in annual %
q = 0.02 # annual dividend rate
sigma = 0.25 # annual volatility in %
steps = 100 # time steps
N = 1000 # number of trials

paths= geo_paths(S,T,r, q,sigma,steps,N)


# In[ ]:


plt.plot(paths);
plt.xlabel("Time Increments")
plt.ylabel("Stock Price")
plt.title("Geometric Brownian Motion")


# In[ ]:


import numpy as np                 #To work with arrays
from datetime import datetime      #To work with our stock data
import pandas_datareader as pdr    #Collects stock data
from scipy.stats import norm 


# In[20]:


import numpy as np                 #To work with arrays
from datetime import datetime      #To work with our stock data
import pandas_datareader as pdr    #Collects stock data
from scipy.stats import norm 


# In[21]:


import numpy as np                 #To work with arrays
from datetime import datetime      #To work with our stock data
import pandas_datareader as pdr    #Collects stock data
from scipy.stats import norm 


# In[28]:


import numpy as np
import matplotlib.pyplot as plt


def geo_paths(S, T, r, q, sigma, steps, N):
    """
    Inputs
    #S = Current stock Price
    #K = Strike Price
    #T = Time to maturity 1 year = 1, 1 months = 1/12
    #r = risk free interest rate
    #q = dividend yield
    # sigma = volatility 
    
    Output
    # [steps,N] Matrix of asset paths 
    """
    dt = T/steps
    #S_{T} = ln(S_{0})+\int_{0}^T(\mu-\frac{\sigma^2}{2})dt+\int_{0}^T \sigma dW(t)
    ST = np.log(S) +  np.cumsum(((r - q - sigma**2/2)*dt +                              sigma*np.sqrt(dt) *                               np.random.normal(size=(steps,N))),axis=0)
    
    return np.exp(ST)

S = 100 #stock price S_{0}
K = 110 # strike
T = 1/2 # time to maturity
r = 0.05 # risk free risk in annual %
q = 0.02 # annual dividend rate
sigma = 0.25 # annual volatility in %
steps = 100 # time steps
N = 12 # number of trials

paths= geo_paths(S,T,r, q,sigma,steps,N)
plt.plot(paths);
plt.xlabel("Time Increments")
plt.ylabel("Stock Price")
plt.title("Time vs Stock Price")


# In[31]:


import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import time
import math

#Plot number of points vs fractional error of pi)
def piPlot(piList):
    
    #Set correct plot size
    piAx = plt.gca() 
    piAx.set_aspect(1)
    
    #Generate and label plot, set to logarithmic scale
    plt.scatter(N,piList)
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Fractional Error vs Number of Points')
    plt.xlabel('Number of Points [Logarithmic]')
    plt.ylabel('Fractional Error [Logarithmic]')
    
#Plot generator for number of points vs processing time
def timePlot(timeList):
    
    #Set correct plot size
    timeAx = plt.gca() 
    timeAx.set_aspect(1)    
    
    #Generate and label plot, set to logarithmic scale
    plt.scatter(N,timeList)
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Processing Time vs Number of Points')
    plt.xlabel('Number of Points [Logarithmic]')
    plt.ylabel('Processing Time [Logarithmic]')

def simulation(n, t):
    #Begin timer
    startTime = t
    
    #Initialize list for current run
    run = []
    
    #Generate random points, find number inside circle
    xPoint = np.random.rand(n)
    yPoint = np.random.rand(n)
    total = ((xPoint ** 2) + (yPoint ** 2) <= 1).sum()
    
    #Estimate pi
    pi = 4*(total/n)
    run.append(pi)
    
    #End timer, calculate total processing itme
    endTime = time.perf_counter()
    T = endTime - startTime
    run.append(T)  

    return run
    
    
def main():
    #Define the total numbers of points for each simulation
    N = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    
    #Run the simulation and calculate fractional error for returned pi
    results1 = [simulation(i, time.perf_counter()) for i in N]
    pis1, times1 = map(list, zip(*results1)) 
    error1 = [abs((math.pi - i)/math.pi) for i in pis1]
    
    #Generate plots
    plot1 = plt.figure(1)
    piPlot(error1)
    
    plot2 = plt.figure(2)
    timePlot(times1)


# In[30]:


N = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 1000000, 100000, 10000, 1000, 100, 10, 1]


# In[32]:


import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import time
import math

#Plot number of points vs fractional error of pi)
def piPlot(piList):
    
    #Set correct plot size
    piAx = plt.gca() 
    piAx.set_aspect(1)
    
    #Generate and label plot, set to logarithmic scale
    plt.scatter(N,piList)
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Fractional Error vs Number of Points')
    plt.xlabel('Number of Points [Logarithmic]')
    plt.ylabel('Fractional Error [Logarithmic]')
    
#Plot generator for number of points vs processing time
def timePlot(timeList):
    
    #Set correct plot size
    timeAx = plt.gca() 
    timeAx.set_aspect(1)    
    
    #Generate and label plot, set to logarithmic scale
    plt.scatter(N,timeList)
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Processing Time vs Number of Points')
    plt.xlabel('Number of Points [Logarithmic]')
    plt.ylabel('Processing Time [Logarithmic]')

def simulation(n, t):
    #Begin timer
    startTime = t
    
    #Initialize list for current run
    run = []
    
    #Generate random points, find number inside circle
    xPoint = np.random.rand(n)
    yPoint = np.random.rand(n)
    total = ((xPoint ** 2) + (yPoint ** 2) <= 1).sum()
    
    #Estimate pi
    pi = 4*(total/n)
    run.append(pi)
    
    #End timer, calculate total processing itme
    endTime = time.perf_counter()
    T = endTime - startTime
    run.append(T)  

    return run
    
    
def main():
    #Define the total numbers of points for each simulation
    N = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    
    #Run the simulation and calculate fractional error for returned pi
    results1 = [simulation(i, time.perf_counter()) for i in N]
    pis1, times1 = map(list, zip(*results1)) 
    error1 = [abs((math.pi - i)/math.pi) for i in pis1]
    
    #Generate plots
    plot1 = plt.figure(1)
    piPlot(error1)
    
    plot2 = plt.figure(2)
    timePlot(times1)


# In[33]:


N = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 1000000, 100000, 10000, 1000, 100, 10, 1]


# In[34]:


import numpy as np
import matplotlib.pyplot as plt


def geo_paths(S, T, r, q, sigma, steps, N):
    """
    Inputs
    #S = Current stock Price
    #K = Strike Price
    #T = Time to maturity 1 year = 1, 1 months = 1/12
    #r = risk free interest rate
    #q = dividend yield
    # sigma = volatility 
    
    Output
    # [steps,N] Matrix of asset paths 
    """
    dt = T/steps
    #S_{T} = ln(S_{0})+\int_{0}^T(\mu-\frac{\sigma^2}{2})dt+\int_{0}^T \sigma dW(t)
    ST = np.log(S) +  np.cumsum(((r - q - sigma**2/2)*dt +                              sigma*np.sqrt(dt) *                               np.random.normal(size=(steps,N))),axis=0)
    
    return np.exp(ST)

S = 100 #stock price S_{0}
K = 110 # strike
T = 1/2 # time to maturity
r = 0.05 # risk free risk in annual %
q = 0.02 # annual dividend rate
sigma = 0.25 # annual volatility in %
steps = 100 # time steps
N = 12 # number of trials

paths= geo_paths(S,T,r, q,sigma,steps,N)
plt.plot(paths);
plt.xlabel("Time Increments")
plt.ylabel("Stock Price")
plt.title("Time vs Stock Price")


# In[35]:


import numpy as np
from scipy.optimize import fsolve, minimize, root
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import math

def compute_CO(s, d, m, k, d_, Rti, Rtd, SigmaTi, SigmaTd, DeltaTi, DeltaTd):

  h1 = (np.log((s + d)/(k + d_))+(Rti - 0.5*SigmaTi**2) * DeltaTi) / (SigmaTi*np.sqrt(DeltaTi))

  h2 = (np.log((s+d)/m)+(Rti-0.5*SigmaTd**2)*DeltaTd) / (SigmaTd*np.sqrt(DeltaTd))

  m1 = np.array([[(h1 + SigmaTi)], [(h2 + SigmaTd)]])
  m2 = np.array([[h1], [h2]])

  rho = np.sqrt(DeltaTi/DeltaTd)
  covariance = np.array([[1, rho],[rho,1]])
  n2 = mvn(cov = covariance)


  call = (s+d)*n2.cdf(np.array([h1+SigmaTi, h2+SigmaTd])) - m*np.exp(-Rtd*DeltaTd)*n2.cdf(np.array([h1,h2])) - k*np.exp(-Rti*DeltaTi)*norm.cdf(h1)

  return call


# In[36]:


from pde import AmericanPut

class GreeksException(Exception):
    pass

class Greeks(object):

    def __init__(self, pricer):
        self.pricer = pricer
        # compute dS
        self.S = S = pricer.S
        self.dS = dS = S[1:]-S[:-1]
        self.dS_minus_half = dS[:-1]
        self.dS_plus_half = dS[1:]
        # compute dU
        self.U = U = pricer.soln
        self.dU = dU = U[1:]-U[:-1]
        self.dU_minus_half = dU[:-1]
        self.dU_plus_half = dU[1:]

    def delta(self, position='central'):
        try:
            d = dict(central  = lambda : self.dU / self.dS,
                     forward  = lambda : self.dU_plus_half / self.dS_plus_half,
                     backward = lambda : self.dU_minus_half / self.dS_minus_half)[position]
        except KeyError:
            errmsg = "delta position argument must be one of ['central', 'forward', 'backward']"
            raise GreeksException(errmsg)
            print errmsg        # XXX log.error instead of stdout print
        else:
            return d()

    def gamma(self):
        d1 = self.delta('forward')
        d2 = self.delta('backward')
        c  = .5 * (self.dS_plus_half + self.dS_minus_half)
        return (d1-d2)/c

##
## quick test
##

def test():

    # BS parameters
    rate             = .01
    sigma            = .3
    strike           = 100.
    init_value       = 100.
    time_to_maturity = 1.

    # compute option object
    option = AmericanPut(rate, sigma, strike, init_value, time_to_maturity)
    option.solve(-1)    # use fully implicit scheme

    # compute greeks
    g = Greeks(option)
    print g.delta('forward')
    print g.gamma()

if __name__ == '__main__':
    test()


# In[ ]:




