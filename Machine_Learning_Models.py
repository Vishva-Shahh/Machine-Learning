"""
Implemented Basic machine learning models in python to get a grasp of how different models perform
Models built - Logistic regression, LDA, KNN, SVC, Decision tree
k-fold validation has also been performed for every model
Neural network with 3 layers, activation function as relu, optimizer as 'adam' and loss as cross entropy is also built
"""

import warnings

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

...

# Load Dataset:

Data = pd.read_csv("bank.csv", delimiter=";")

# Describing/Understanding Data

print(Data.isnull())
print(Data.shape)
print(Data.head(20))
print(Data.describe())
print(Data.dtypes)

"""
We have some categorical variables in our dataset on which we will perform one-hot encoding.

The following variables require the mentioned coding:
1. Marital
2. default
3. housing
4. loan
5. y
6. job
7. education
8. contact
9. month
10. poutcome
"""

# There is a mechanical way of doing this and a smart way of doing this:

# First we try the mechanical way as follows:

"""
We do categorical hard-coding: 
Eg: "no" as 0 and "yes as 1 in column loan and y 
and "single" as 0, "married" as 1 and "divorced" as 2 in column marital
"""

replace_married = {'marital': {"single": 0, "married": 1, "divorced": 2}}
Data.replace(replace_married, inplace=True)

Data['y'] = np.where(Data['y'].str.contains('yes'), 1, 0)
Data['loan'] = np.where(Data['loan'].str.contains('yes'), 1, 0)

"""
Now, we will do the smart way of coding:
"""

# First we find how many are dtype - objects

cat_columns = Data.select_dtypes(['object']).columns

print(cat_columns)

# Then we convert them to categorical variables

Data[cat_columns] = Data[cat_columns].astype('category')

# Now, we use the apply function and code all at the same time:

Data[cat_columns] = Data[cat_columns].apply(lambda x: x.cat.codes)

print(Data.dtypes)

"""
Our data is now processed. Now we can build model on this dataset. 
"""

# Splitting data into train, validation and test:

array = Data.values
X = array[:, 0:15]
Y = array[:, 16]
validation_size = 0.20
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

# We can train all the models at once or do it individually:

models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()), ('SVM', SVC())]

# evaluate each model in turn
results = []
names = []
msgs = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    y_pred = cross_val_predict(model, X, Y, cv=10)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f" % (name, cv_results.mean())
    msgs.append(msg)
    conf_matrix = confusion_matrix(Y, y_pred)
    print(msgs)
    print(conf_matrix)

print("The best model with highest accuracy is "+str(max(msgs)))


# In[2]:

# Also can be done individually by:

# Logistic Regression

clf = make_pipeline(MinMaxScaler(), LogisticRegression())
y_pred_1 = cross_val_predict(clf, X, Y, cv=10)
conf_mat = confusion_matrix(Y, y_pred_1, labels=[0, 1])
print("Accuracy is", accuracy_score(Y, y_pred_1))
print(conf_mat)

# Decision Tree

DT = make_pipeline(MinMaxScaler(), DecisionTreeClassifier())
y_pred_2 = cross_val_predict(DT, X, Y, cv=10)
conf_mat_dt = confusion_matrix(Y, y_pred_2, labels=[0, 1])
print("Accuracy is", accuracy_score(Y, y_pred_2))
print(conf_mat_dt)

# KNN

DT = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
y_pred_3 = cross_val_predict(DT, X, Y, cv=10)
conf_mat_KNN = confusion_matrix(Y, y_pred_3, labels=[0, 1])
print("Accuracy is", accuracy_score(Y, y_pred_3))
print(conf_mat_KNN)

# SVC

DT = make_pipeline(MinMaxScaler(), SVC())
y_pred_4 = cross_val_predict(DT, X, Y, cv=10)
conf_mat_SVM = confusion_matrix(Y, y_pred_4, labels=[0, 1])
print("Accuracy is", accuracy_score(Y, y_pred_4))
print(conf_mat_SVM)

# In[3]:

# Neural Networks:

model = Sequential()
model.add(Dense(12, input_dim=15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy * 100))

predictions = model.predict_classes(X)

for i in range(len(X)):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
