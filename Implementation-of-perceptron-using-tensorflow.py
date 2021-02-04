#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import zipfile
import numpy as np
import pandas as pd


# In[ ]:


import os
import zipfile
import numpy as np
import pandas as pd


# In[ ]:


df=pd.read_csv("/content/sonar.all-data.csv")


# In[ ]:


df.head()


# In[ ]:


df['R'].value_counts()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


dataset = df.values


# In[ ]:


len(df)


# In[ ]:


dataset


# In[ ]:


len(dataset)


# In[ ]:


seed = 7
np.random.seed(seed)


# In[ ]:


df.shape


# In[ ]:


import tensorflow as tf


# In[ ]:


import keras


# In[ ]:



from keras import preprocessing


# In[ ]:


from sklearn import preprocessing


# In[ ]:


df.head()
le=preprocessing.LabelEncoder()
for i in range(len(df.columns)-1,len(df.columns)):
  df.iloc[:,i]=le.fit_transform(df.iloc[:,i]).astype(float)

df.head()


# In[ ]:



df['R'].value_counts()


# In[ ]:


x=df.drop(['R'],axis=1)
y=df['R']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)


# In[ ]:


clf.fit(x, y)


# In[ ]:


from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


# In[ ]:


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=200):
        self.n_iter = n_iter
        self.eta = eta
    def fit(self,x,y, chooseWeightVector, x_test,y_test):
        if chooseWeightVector == 1:
            self.w_ = np.random.rand(1 + x.shape[1])
        else:
            self.w_ = np.zeros(1 + x.shape[1])
        self.w_[0] = 1        
        self.errors_ = []
        self.accuracies_ = []
        
        for _ in range(self.n_iter):
            #zip: Make an iterator that aggregates elements from each of the iterables.
            for xi, target in zip(x, y):
                # w <- w + α(y — f(x))x or alternately
                # w <- w + α(t - o)x
                # predict is: o = sign(w * x) != t
                o = self.predict(xi)
                update = self.eta * (target - o)
                self.w_[1:] += update * xi
                self.w_[0] += update
            self.calc_error(x_test,y_test)
            
    
    def calc_error(self, x, y):
        errors = 0
        sumOfAccuracy = 0
        for x_t, y_t in zip(x,y):
            y_pred = self.predict(x_t)
            errors += np.square(y_t-y_pred)
            sumOfAccuracy += 1 if y_pred == y_t else 0
        self.errors_.append(errors/(2*len(x)))
        self.accuracies_.append(sumOfAccuracy/len(x))
    
    def net_input(self,x):
        # sum(wi * xi)
        # w · x + b
        return np.dot(x, self.w_[1:]) + self.w_[0]
    def predict(self, x):
        #sign(net)
        return np.where(self.net_input(x) >= 0.0, 1, -1)


# In[ ]:


#try different learning rate
learning_rate = [0.005,0.05,0.5]
for i in learning_rate:
    p_null = Perceptron(i)
    p_null.fit(x_train.values,y_train, 0, x_test.values, y_test)
    #p_null.fit(X_train.values,y_train)
    #print(p_null.score(X_test.values,y_test))
    
    #plot cost function
    plt.plot(range(1, len(p_null.errors_) + 1), p_null.errors_, marker='o')
    plt.title('learning_rate = {}'.format(i))
    plt.xlabel('Epochs')
    plt.ylabel('Cost function')
    #plt.savefig('images/02_07.png', dpi=300)
    plt.show()
    
    #plot accuracy in every epoch
    plt.plot(range(1, len(p_null.accuracies_) + 1), p_null.accuracies_)
    plt.title('learning_rate = {}'.format(i))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.savefig('images/02_07.png', dpi=300)
    plt.show()
    
    print('Mean accuracy on test set with {} epochs and learning rate={}: {} '.format(p_null.n_iter,i,sum(p_null.accuracies_)/(len(p_null.accuracies_))))


# In[ ]:


x_test.head()


# In[ ]:


y_pred=clf.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


mean_squared_error(y_test,y_pred)


# In[ ]:


MSE=np.square(np.subtract(y_test,y_pred)).mean()


# In[ ]:


MSE


# In[ ]:


from sklearn.metrics import precision_score


# In[ ]:


precision_score(y_test, y_pred)


# In[ ]:




