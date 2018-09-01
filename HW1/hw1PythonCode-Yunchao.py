
# coding: utf-8

# In[1]:


import pandas as pd
train_df = pd.read_csv('./Titanic/train.csv')
test_df = pd.read_csv('./Titanic/test.csv')
combine = [train_df, test_df]


# In[2]:


train_df.head()


# In[3]:


train_df.info()


# In[4]:


train_df[["Age","SibSp","Parch","Fare"]].describe()


# In[5]:


test_df[["Age","SibSp","Parch","Fare"]].describe()


# In[6]:


train_df.describe(include=['O'])


# In[7]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[8]:


train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[9]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[10]:


g = sns.FacetGrid(train_df, col='Survived', row='Pclass', aspect=1.5)
g.map(plt.hist, 'Age')
g.add_legend()


# In[11]:


g = sns.FacetGrid(train_df, row='Embarked', col='Survived', aspect=1.5)
g.map(sns.barplot, 'Sex', 'Fare')
g.add_legend()


# In[12]:


train_df.describe(include=['O'])


# In[13]:


test_df.describe(include=['O'])


# In[14]:


train_df = train_df.drop(['Ticket', 'Cabin', 'Name','PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Name','PassengerId'], axis=1)
combine = [train_df, test_df]


# In[15]:


for data in combine:
    data['Gender'] = data['Sex'].map({'female':1, 'male':0}).astype(int)
    
train_df.tail()


# In[16]:


import numpy as np

for data in combine:
    data.loc[data.Age.isnull(), 'Age'] = round(np.random.normal(29.699118, 14.526497),1)

train_df.tail()


# In[17]:


train_df.tail()


# In[18]:


mean_ages = np.zeros((3,2))
std_ages = np.zeros((3,2))
for dataset in combine:

    for j in range(0, 3):
        for k in range(0,2):
            age_df = dataset[(dataset['Pclass'] == j+1) &                                (dataset['Gender'] == k)]['Age'].dropna()

            mean_ages[j,k] = int(age_df.mean()/0.5 + 0.5)*0.5
            std_ages[j,k] = int(age_df.std()/0.5 + 0.5)*0.5


    for j in range(0, 3):
        for k in range(0,2):
            dataset.loc[ (dataset.Age.isnull())  & (dataset.Pclass == j+1) & (dataset.Gender == k),                'Age'] = np.random.normal(mean_ages[j,k],std_ages[j,k])

    dataset['Age'] = dataset['Age'].astype(int)


# In[19]:


for data in combine:
    data.loc[data.Embarked.isnull(), 'Embarked'] = 'S'


# In[20]:


for data in combine:
    data.loc[data.Fare.isnull(), 'Fare'] = 32.204208


# In[24]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

