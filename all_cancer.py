#!/usr/bin/env python
# coding: utf-8

# In[1]:


data = pd.read_csv("/home/ashraf/projects/privacy/updated.csv")


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[3]:


data = pd.read_csv("/home/ashraf/projects/privacy/updated.csv")


# In[4]:


len(data.index), len(data.columns)


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.isna()


# In[8]:


data.isna().any()


# In[9]:


data.isna().sum() 


# In[10]:


data = data.dropna(axis='columns')


# In[11]:


data.describe(include="O")


# In[12]:


data.diagnosis.value_counts()


# In[13]:


diagnosis_unique = data.diagnosis.unique()


# In[14]:


diagnosis_unique


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[30]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[33]:


import plotly.express as px
import plotly.graph_objects as go

# plt.figure(figsize=(7,12))
px.histogram(data, x='diagnosis')
# plt.show()


# In[34]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[35]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[36]:


from sklearn.preprocessing import LabelEncoder


# In[37]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[38]:


data.head(2)


# In[39]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[40]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[51]:


plt.figure(figsize=(10, 8))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[42]:


plt.figure(figsize=(15, 10))


fig = px.imshow(data[cols].corr());
fig.show()


# In[52]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[53]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[54]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[55]:


data.columns


# In[56]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[57]:


X = data[prediction_feature]
X

# print(X.shape)
# print(X.values)


# In[58]:


y = data.diagnosis
y

# print(y.values)


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)
# print(X_test)


# In[67]:


# Scale the data to keep all the values in the same magnitude of 0 -1 

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[68]:


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


# In[69]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[63]:


print(list(models_list.keys()))
print(list(models_list.values()))

# print(zip(list(models_list.keys()), list(models_list.values())))


# In[70]:


# Let's Define the function for confision metric Graphs

def cm_metrix_graph(cm):
    
    sns.heatmap(cm,annot=True,fmt="d")
    plt.show()
        


# In[71]:


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


# In[72]:


print(len(confusion_matrixs))


# In[73]:


df_pred


# In[74]:


data = pd.read_csv("/home/ashraf/projects/privacy/synthetic data.csv")


# In[75]:


len(data.index), len(data.columns)


# In[76]:


data.head()


# In[77]:


data.info()


# In[78]:


data.isna()


# In[79]:


data.isna().any()


# In[80]:


data.isna().sum() 


# In[81]:


data = data.dropna(axis='columns')


# In[82]:


data.describe(include="O")


# In[83]:


data.diagnosis.value_counts()


# In[84]:


diagnosis_unique = data.diagnosis.unique()


# In[85]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[87]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[88]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[89]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[90]:


from sklearn.preprocessing import LabelEncoder


# In[91]:


data.head(2)


# In[92]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[93]:


data.head(2)


# In[94]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[95]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[96]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[97]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[98]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[99]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[100]:


data.columns


# In[101]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[102]:


X = data[prediction_feature]
X


# In[103]:


y = data.diagnosis
y


# In[104]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)


# In[105]:


# Scale the data to keep all the values in the same magnitude of 0 -1 

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[106]:


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


# In[107]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[108]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[109]:


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


# In[112]:


df_pred


# In[113]:


df_pred


# In[114]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[115]:


data = pd.read_csv("/home/ashraf/projects/privacy/Breast_cancer_data.csv")


# In[116]:


len(data.index), len(data.columns)


# In[117]:


data.shape


# In[118]:


data.head()


# In[119]:


data.info()


# In[120]:


data.isna()


# In[121]:


data.isna().any()


# In[122]:


data.isna().sum() 


# In[123]:


data = data.dropna(axis='columns')


# In[124]:


data.describe(include="O")


# In[125]:


data.diagnosis.value_counts()


# In[126]:


data.head(2)


# In[127]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[128]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[129]:


cols = ["diagnosis", "mean_radius", "mean_texture", "mean_perimeter", "mean_area","mean_smoothness"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[130]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[131]:


data = pd.read_csv("/home/ashraf/projects/privacy/dataR2.csv")


# In[132]:


len(data.index), len(data.columns)


# In[133]:


data.shape


# In[134]:


data.head()


# In[135]:


data.info()


# In[136]:


data.isna()


# In[137]:


data.isna().any()


# In[138]:


data.isna().sum() 


# In[139]:


data.Classification.value_counts()


# In[140]:


data.head(2)


# In[141]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[143]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist( data.Classification)
# plt.legend()
plt.title("Counts of Classification")
plt.xlabel("Classification")


plt.subplot(1, 2, 2)

sns.countplot('Classification', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show()


# In[144]:


cols = ["Age", "BMI", "Glucose", "Insulin", "HOMA","Leptin","Adiponectin","Resistin","MCP.1","Classification"]

sns.pairplot(data[cols], hue="Classification")
plt.show()


# In[145]:


from sklearn.preprocessing import LabelEncoder


# In[146]:


data.head(2)


# In[148]:


labelencoder_Y = LabelEncoder()
data.Classification = labelencoder_Y.fit_transform(data.Classification)


# In[149]:


data.head(2)


# In[150]:


print(data.Classification.value_counts())
print("\n", data.Classification.value_counts().sum())


# In[151]:


cols = ['Classification', 'Age', 'BMI', 'Glucose',
       'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']
print(len(cols))
data[cols].corr()


# In[152]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[153]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[154]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[155]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[157]:


prediction_feature = ['Age', 'BMI', 'Glucose',
       'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']

targeted_feature = 'Classification'

len(prediction_feature)


# In[158]:


X = data[prediction_feature]
X


# In[159]:


y = data.Classification
y


# In[160]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)


# In[161]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[162]:


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


# In[163]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[164]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[165]:


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


# In[167]:


df_pred


# In[168]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[169]:


data = pd.read_csv("/home/ashraf/projects/privacy/third.csv")


# In[170]:


data.shape


# In[171]:


data.head()


# In[172]:


data.info()


# In[173]:


data.isna()


# In[174]:


data.isna().any()


# In[175]:


data.isna().sum() 


# In[176]:


data.describe(include="O")


# In[177]:


data.Type.of.Death.value_counts()


# In[178]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[179]:


data = pd.read_csv("/home/ashraf/projects/privacy/second_dataR2.csv")


# In[180]:


len(data.index), len(data.columns)


# In[181]:


data.shape


# In[182]:


data.head()


# In[183]:


data.tail()


# In[184]:


data.info()


# In[185]:


data.isna()


# In[186]:


data.isna().any()


# In[187]:


data.isna().sum() 


# In[188]:


data.Classification.value_counts()


# In[189]:


data.head(2)


# In[190]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[191]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist( data.Classification)
# plt.legend()
plt.title("Counts of Classification")
plt.xlabel("Classification")


plt.subplot(1, 2, 2)

sns.countplot('Classification', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show()


# In[192]:


cols = ["Age", "BMI", "Glucose", "Insulin", "HOMA","Leptin","Adiponectin","Resistin","MCP.1","Classification"]

sns.pairplot(data[cols], hue="Classification")
plt.show()


# In[193]:


from sklearn.preprocessing import LabelEncoder


# In[194]:


data.head(2)


# In[195]:


labelencoder_Y = LabelEncoder()
data.Classification = labelencoder_Y.fit_transform(data.Classification)


# In[196]:


data.head(2)


# In[197]:


print(data.Classification.value_counts())
print("\n", data.Classification.value_counts().sum())


# In[198]:


cols = ['Classification', 'Age', 'BMI', 'Glucose',
       'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']
print(len(cols))
data[cols].corr()


# In[199]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[200]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[201]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[202]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[203]:


prediction_feature = ['Age', 'BMI', 'Glucose',
       'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']

targeted_feature = 'Classification'

len(prediction_feature)


# In[204]:


X = data[prediction_feature]
X


# In[205]:


y = data.Classification
y


# In[206]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)


# In[207]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[208]:


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


# In[209]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[210]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[211]:


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


# In[213]:


df_pred


# In[214]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[215]:


data = pd.read_csv("/home/ashraf/projects/privacy/1.csv")


# In[216]:


len(data.index), len(data.columns)


# In[217]:


data.head()


# In[218]:


data
.
head
(
)


# In[219]:


data.info()


# In[220]:


data.isna()


# In[238]:


data.isna().any()


# In[222]:


data.isna().sum() 


# In[223]:


data = data.dropna(axis='columns')


# In[224]:


data.describe(include="O")


# In[225]:


data.diagnosis.value_counts()


# In[226]:


diagnosis_unique = data.diagnosis.unique()


# In[227]:


diagnosis_unique


# In[228]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[229]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[230]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[231]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[232]:


from sklearn.preprocessing import LabelEncoder


# In[233]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[234]:


data.head(2)


# In[235]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[236]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[237]:


plt.figure(figsize=(10, 8))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[239]:


data.info()


# In[284]:


data = pd.read_csv("/home/ashraf/projects/privacy/1.csv")


# In[285]:


len(data.index), len(data.columns)


# In[286]:


data.shape


# In[287]:


data.head()


# In[244]:


data.info()


# In[245]:


data.isna()


# In[246]:


data.isna().any()


# In[247]:


data.isna().sum() 


# In[248]:


data.describe(include="O")


# In[249]:


data.diagnosis.value_counts()


# In[250]:


data.head(2)


# In[251]:


diagnosis_unique = data.diagnosis.unique()


# In[252]:


diagnosis_unique


# In[253]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[254]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[255]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[256]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[257]:


from sklearn.preprocessing import LabelEncoder


# In[258]:


data.head(2)


# In[259]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[260]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[261]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[262]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[263]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[264]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[265]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[266]:


data.columns


# In[267]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[268]:


X = data[prediction_feature]
X


# In[269]:


y = data.diagnosis
y


# In[270]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)


# In[271]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[272]:


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


# In[273]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[274]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[275]:


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


# In[276]:


def cm_metrix_graph(cm):
    
    sns.heatmap(cm,annot=True,fmt="d")
    plt.show()


# In[277]:


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


# In[278]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[279]:


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


# In[280]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[281]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[282]:


# Let's Define the function for confision metric Graphs

def cm_metrix_graph(cm):
    
    sns.heatmap(cm,annot=True,fmt="d")
    plt.show()
        


# In[283]:


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


# In[288]:


data.info()


# In[289]:


data.isna()


# In[290]:


data.isna().any()


# In[291]:


data.isna().sum() 


# In[292]:


data = data.dropna(axis='columns')


# In[293]:


data.describe(include="O")


# In[294]:


data.head(2)


# In[295]:


diagnosis_unique = data.diagnosis.unique()


# In[296]:


diagnosis_unique


# In[297]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[298]:


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist( data.diagnosis)
# plt.legend()
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")


plt.subplot(1, 2, 2)

sns.countplot('diagnosis', data=data); # ";" to remove output like this > <matplotlib.axes._subplots.AxesSubplot at 0x7f3a1dddba50>

# plt.show() 


# In[299]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis")
plt.show()


# In[300]:


size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# In[301]:


from sklearn.preprocessing import LabelEncoder


# In[302]:


data.head(2)


# In[303]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[304]:


data.head(2)


# In[305]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# In[306]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[307]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
     'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[308]:


plt.figure(figsize=(10, 8))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# In[309]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# In[310]:


from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier


# In[311]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# In[312]:


data.columns


# In[314]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[315]:


X = data[prediction_feature]
X


# In[316]:


y = data.diagnosis
y


# In[317]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

print(X_train)
# print(X_test)


# In[318]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[319]:


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


# In[320]:


models_list = {
    "LogisticRegression" :  LogisticRegression(),
    "RandomForestClassifier" :  RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier" :  DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC" :  SVC(),
}


# In[321]:


print(list(models_list.keys()))
print(list(models_list.values()))


# In[322]:


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


# In[323]:


df_pred


# In[324]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[325]:


data = pd.read_csv("/home/ashraf/projects/privacy/2.csv")


# In[326]:


len(data.index), len(data.columns)


# In[327]:


data.head()


# In[ ]:




