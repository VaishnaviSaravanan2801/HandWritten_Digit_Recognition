
# coding: utf-8

# # Project 2 DA using Scikit_learn Coding Internship 

# ## Topic: Recognizing Handwritten Digits with scikit-learn

# ## Problem Definition

# 
# Recognizing handwritten text is a problem that can be traced back to the first automatic machines that needed to recognize individual characters in handwritten documents.To address this issue in Python, the scikit-learn library provides a good example to better understand this technique, the issues involved, and the possibility of making predictions.Here we are predicting a numeric value, and then reading and interpreting an image that uses a handwritten font.

# ### Import the model

# In[1]:


from sklearn import svm


# ### Creating Instance

# In[2]:


svc = svm.SVC(gamma=0.001, C=100.)


# ### Load Dataset

# In[3]:


from sklearn import datasets
digits = datasets.load_digits()
print(digits)


# ## Data Extraction

# In[4]:


print(digits.DESCR)


# In[5]:


digits.images[0]


# ## Data preparation - Data transformation

# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')


# ## Data Preparation - Analyze & split

# In[7]:


dir(digits)


# In[8]:


print(type(digits.images))
print(type(digits.target))


# In[9]:


digits.images.shape


# In[10]:


digits.target


# In[11]:


digits.target.shape


# In[12]:


digits.target.size


# ### Data Visualization

# In[13]:


def plot_multi(i):
    #'''Plots 10 digits, starting with digit i'''
    nplots = 10
    fig = plt.figure(figsize=(9,9))
    for j in range(nplots):
        plt.subplot(4,5,j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    plt.show()
plot_multi(0)


# ### Case 1: 

# ### Range of training set: [401:1790] & validation set: [1791:1796] 

# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplot(321)
plt.imshow(digits.images[1791], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(322)
plt.imshow(digits.images[1792], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(323)
plt.imshow(digits.images[1793], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(324)
plt.imshow(digits.images[1794], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(325)
plt.imshow(digits.images[1795], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(326)
plt.imshow(digits.images[1796], cmap=plt.cm.gray_r, interpolation='nearest')


# ### Case 2:

# ### Range of training set: [321:800] & validation set: [801:806] 

# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplot(321)
plt.imshow(digits.images[801], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(322)
plt.imshow(digits.images[802], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(323)
plt.imshow(digits.images[803], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(324)
plt.imshow(digits.images[804], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(325)
plt.imshow(digits.images[805], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(326)
plt.imshow(digits.images[806], cmap=plt.cm.gray_r, interpolation='nearest')


# ### Case 3:

# ### Range of training set: [1:260] & validation set: [261:266] 

# In[16]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplot(321)
plt.imshow(digits.images[261], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(322)
plt.imshow(digits.images[262], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(323)
plt.imshow(digits.images[263], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(324)
plt.imshow(digits.images[264], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(325)
plt.imshow(digits.images[265], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(326)
plt.imshow(digits.images[266], cmap=plt.cm.gray_r, interpolation='nearest')


# ## Training the instance

# ### Case 1:

# In[17]:


svc.fit(digits.data[1:1790], digits.target[1:1790])


# ### Case 2:

# In[18]:


svc.fit(digits.data[321:800], digits.target[321:800])


# ### Case 3:

# In[19]:


svc.fit(digits.data[1:260], digits.target[1:260])


# ### Train-test split

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)


# In[23]:


svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
y_pred


# ## Model validation/test

# ### Case 1:

# In[24]:


svc.predict(digits.data[1791:1797])


# In[25]:


digits.target[1791:1797]


# ### Case 2:

# In[26]:


svc.predict(digits.data[801:806])


# In[27]:


digits.target[801:806]


# ### Case 3:

# In[28]:


svc.predict(digits.data[261:266])


# In[29]:


digits.target[261:266]


# ### Accuracy

# In[30]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)*100
print(accuracy)


# ## Deploy - Visualization and Interpretation of results

# In[32]:


import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,y_pred)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Pastel1');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15);


# 
# The svc estimator has learned correctly. It is able to recognize the handwritten digits, interpreting more or less correctly all six digits of the validation set.

# 
# On choosing a smaller training set and different range for validation, I analyzed that

# #### Case 1: We have got 100% accurate prediction 

# #### Case 2: We have got 100% accurate prediction 

# #### Case 3: We have got 95% accurate prediction
