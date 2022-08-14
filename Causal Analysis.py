#!/usr/bin/env python
# coding: utf-8

# Causal Analysis
# ---
# 
# 
# Author - David H.

# ##### Causal Analysis is an experimental analysis within the statistical field to establish cause and effect.

# #### Import the required libraries

# In[1]:


import os
import glob

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')


# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase

# Scipy
from scipy import interpolate
from scipy import spatial
from scipy import stats
from scipy.cluster import hierarchy
# Others
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import pickle
from math import modf

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn import metrics
import xgboost as xgb
from sklearn.linear_model import SGDClassifier


sns.set(style='white', context='notebook', palette='deep')
from sklearn.model_selection import train_test_split, KFold, cross_validate

#Models
import warnings
warnings.filterwarnings("ignore")


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedShuffleSplit, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from sklearn.inspection import plot_partial_dependence

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier  
from sklearn.exceptions import ConvergenceWarning

pd.options.display.float_format = '{:.5f}'.format
import numpy as np
import matplotlib.pyplot as plt
plt.style.context('seaborn-talk')
plt.style.use('fivethirtyeight')
params = {'legend.fontsize': '16',
          'figure.figsize': (15, 5),
         'axes.labelsize': '20',
         'axes.titlesize':'30',
         'xtick.labelsize':'18',
         'ytick.labelsize':'18'}
plt.rcParams.update(params)

plt.rcParams['text.color'] = '#A04000'
plt.rcParams['xtick.color'] = '#283747'
plt.rcParams['ytick.color'] = '#808000'
plt.rcParams['axes.labelcolor'] = '#283747'

# Seaborn style (figures)
sns.set(context='notebook', style='whitegrid')
sns.set_style('ticks', {'xtick.direction':'in', 'ytick.direction':'in'})
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn import metrics

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score


# In[2]:


observation_window1 = pd.read_csv('new_data/features_201506.csv')
observation_window2 = pd.read_csv('new_data/features_201512.csv')
outcome_window = pd.read_csv('new_data/features_201606.csv')


# In[3]:


print('Shape of Observation Window 1 : ', observation_window1.shape)
print('Shape of Observation Window 2 : ', observation_window2.shape)
print('Shape of Outcome Window 1 : ', outcome_window.shape)


# In[4]:


features = pd.concat([observation_window1, observation_window2], axis=0, ignore_index=True)


# In[5]:


features = features.drop_duplicates()


# In[6]:


features = features[features['acc_balance']>1500]
features = features[features['acc_tenure']>6]
features = features.reset_index(drop = True)
features.drop(['churn'], axis = 1, inplace = True)
print('Shape of Features : ', features.shape)


# In[7]:


outcome_window = outcome_window[['new_id', 'churn']]
df = pd.merge(features, outcome_window, on = 'new_id')


# In[8]:


plt.style.use('fivethirtyeight')
display(df.isna().sum()/len(df)*100)
(df.isna().sum()/len(df)*100).plot(kind = 'barh', figsize = (17,6));
plt.title('Percentage of Missing Values')
plt.show()


# In[9]:


display(df['churn'].value_counts()/len(df)*100)
(df['churn'].value_counts()/len(df)*100).plot(kind = 'barh', figsize = (17,6));
plt.title('Distribution of Churn')
plt.show();


# ### Data Is Imbalanced

# In[10]:


print("Shape of Combined Dataframe : ", df.shape)


# ## Feature Engineering

# #### Dropping Duplicates

# In[11]:


finalDF = df.copy()
finalDF = finalDF.drop_duplicates()
print("Shape of Combined Dataframe : ", finalDF.shape)


# In[12]:


finalDF.drop_duplicates(subset='new_id', keep='last', inplace=True, ignore_index=False)
print("Shape of Combined Dataframe : ", finalDF.shape)


# In[13]:


display(finalDF['churn'].value_counts()/len(finalDF)*100)
(finalDF['churn'].value_counts()/len(finalDF)*100).plot(kind = 'barh', figsize = (17,6));
plt.title('Distribution of Churn')
plt.show();


# In[14]:


finalDF.head()


# ### Causal Analysis with DoWhy
# 
# According to the DoWhy documentation Page, DoWhy is a Python Library that sparks causal thinking and analysis via 4-steps:
# - Model a causal inference problem using assumptions that we create.
# - Identify an expression for the causal effect under these assumptions (“causal estimand”).
# - Estimate the expression using statistical methods such as matching or instrumental variables.
# - Verify the validity of the estimate using a variety of robustness checks.

# If we make it more simple, the way DoWhy package done Causal Analysis is by **Creating Causal Model -> Identify Effect -> Estimate the Effect -> Validate.**

# ### Factors/or Attribution that could affect the Churn most.. fir Hypothesis Testing
# 1. acc_balance
# 2. acc_balance_change_amount
# 3. account_growth
# 4. cust_tenure

# ## **1. Would Account Balance affecting the bank churn?**

# In[15]:


finalDF.acc_balance.hist(figsize = (17,6), bins = 100)
plt.show();

finalDF.acc_balance.describe()


# In[16]:


finalDF.head()


# In[17]:


df = finalDF.copy()


# ## High Account Balance
# 
# ## High Customer Tenure or Age of Bank Account
# 
# ## High Account Balance Change Amount

# ### Creating the High_Limit attribute for Account Balance
# #### If Account balance is greater than 50,000.. then the account is considered as High Account Balance
# 
# #### If Customer Tenure is greater than 10 years.. then the account is considered as High_tenure
# 
# #### If  Account Balance Change Amount is greater than 10000 .. then the account is considered as high_acc_balance_change_amount
# 

# In[18]:


## High Account Balance

finalDF['High_limit'] = finalDF['acc_balance'].apply(lambda x: True if x > 50000 else False)

## High Customer Tenure or Age of Bank Account

finalDF['High_tenure'] = finalDF['cust_tenure'].apply(lambda x: True if x > 10 else False)

# High Account Balance Change Amount
finalDF['acc_balance_change_amount'] = finalDF['acc_balance_change_amount'].abs()


finalDF['High_bal_change'] = finalDF['acc_balance_change_amount'].apply(lambda x: True if x > 10000 else False)


# In[19]:


# Creating True or False columns from the Attrition flag for the churn column
finalDF['Churn'] = finalDF['churn'].apply(lambda x: True if x == 'Attrited Customer' else False)


# ### Create the Causal Model

# In[20]:


training= finalDF[['acc_balance', 'acc_balance_change_amount', 'account_growth', 'cust_tenure', 'channel','age','gender', 'High_limit', 'High_tenure', 'High_bal_change', 'churn' ]].copy()


# From all these features, I have a few assumptions that would affect the churn and other features:
# 
# - The high Limit category might affect the Churn, as people with a lower limit (Account Balance) might not that loyal to the bank compared to customers to high bank account balance.
# - Account Tenure  would affect the Balance high limit. 
# - Gender would affect the income and hence Account Balance and tenure.
# - The customer age might affect the education level they had and the income category.

# In[21]:


#Creating the 
causal_graph = """
digraph {
age;

gender;High_bal_change;
High_limit;
High_tenure;

cust_tenure;
acc_balance;
churn;
acc_balance_change_amount;
age -> acc_balance;
cust_tenure -> acc_balance;
gender -> acc_balance;
age -> acc_balance_change_amount;
age -> High_tenure;
High_limit->churn;
cust_tenure -> High_tenure;
High_limit -> High_bal_change
acc_balance -> High_bal_change
High_bal_change -> churn;
cust_tenure -> churn;
acc_balance -> churn;
}
"""


# Estimate the Causal Effect based on the statistical method
# The treatment's causal effect on the outcome is based on the change in the value of the treatment variable. How strong the effect is a matter of statistical estimation. There are many methods for the statistical estimation of the causal effect, which you could read here for a more clear explanation.
# There are few methods to Estimate the Causal Effect, and that is:
# - Propensity Score Matching
# - Propensity Score Stratification
# - Propensity Score-based Inverse Weighting
# - Linear Regression
# - Generalized Linear Models (e.g., logistic regression)
# - Instrumental Variables
# - Regression Discontinuity
# 
# For the estimation example, I would use the “Propensity Score-Based Inverse Weighting” method./

# # 1. High Account Balance

# In[22]:


from dowhy import CausalModel
from IPython.display import Image, display
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['High_limit'],
        outcome='churn')
model.view_model()


# Above are our Causal model and all the assumptions we already think about. 

# #### Identify the Causal Effect

# Causal Analysis states that the Treatment affecting the Outcome if changing the treatment affects the Outcome when everything else is still the same (constant).
# Using the DoWhy Causal Model, we would identify this Causal Effect.

# In[23]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[24]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[25]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~-0.091, where it is equivalent to saying that the probability of churn is decreased by ~9% when the customer has a higher account balance.

# ### Refute the obtained Estimate
# The causal effect estimation is based on the data's statistical estimation, but the causality itself is not based on the data; rather, it based on our assumptions previously.
# Using the DoWhy package, we could test our assumption validity via multiple robustness checks. These are some of the methods available to test our assumptions:
# - Adding a randomly-generated confounder
# - Adding a confounder that is associated with both treatment and outcome
# - Replacing the treatment with a placebo (random) variable
# - Removing a random subset of the data

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[26]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[27]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# It seems based on the refutal method; we could agree that our assumption was correct that the High Limit had a causal effect on the churn.

# # 2. High Customer Tenure

# In[28]:


from dowhy import CausalModel
from IPython.display import Image, display
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['High_tenure'],
        outcome='churn')
model.view_model()


# In[29]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[30]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[31]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~-0.027, where it is equivalent to saying that the probability of churn is decreased by ~3% when the customer has a higher tenure with bank (greater than 3%).

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[32]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[33]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# # 3. High Balance Change Amount

# In[34]:


from dowhy import CausalModel
from IPython.display import Image, display
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['High_bal_change'],
        outcome='churn')
model.view_model()


# In[35]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[36]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[37]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~-0.1234, where it is equivalent to saying that the probability of churn is decreased by ~12% when the customer has a higher account balance.

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[38]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[39]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# ## Lets consider the High Change with two perspective.
# 1. High Positive Change (Credit)
# 2. High Negative Change (Debit)

# In[40]:


finalDF['acc_balance_change_amount'] = df['acc_balance_change_amount']

finalDF['High_credit_change'] = finalDF['acc_balance_change_amount'].apply(lambda x: True if x > 50000 else False)
finalDF['High_debit_change'] = finalDF['acc_balance_change_amount'].apply(lambda x: True if -x > -50000 else False)


# In[41]:


#Creating the 
causal_graph = """
digraph {
age;

gender;
High_credit_change;
High_debit_change;
High_limit;
High_tenure;

cust_tenure;
acc_balance;
churn;
acc_balance_change_amount;
age -> acc_balance;
cust_tenure -> acc_balance;
gender -> acc_balance;
age -> acc_balance_change_amount;
age -> High_tenure;
High_limit->churn;
cust_tenure -> High_tenure;
High_limit -> High_credit_change
acc_balance -> High_credit_change

High_limit -> High_debit_change
acc_balance -> High_debit_change
High_bal_change -> churn;
cust_tenure -> churn;
acc_balance -> churn;
}
"""


# # 4. High Credit Account Balance Change

# In[42]:


training= finalDF[['acc_balance', 'acc_balance_change_amount', 'account_growth', 'cust_tenure', 'channel','age','gender', 'High_limit', 'High_tenure', 'High_bal_change','High_credit_change', 'High_debit_change','churn']].copy()


# In[43]:


from dowhy import CausalModel
from IPython.display import Image, display
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['High_credit_change'],
        outcome='churn')
model.view_model()


# Above are our Causal model and all the assumptions we already think about. 

# #### Identify the Causal Effect

# In[44]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[45]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[46]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~-0.145, where it is equivalent to saying that the probability of churn is increased by ~14% when the customer has a higher Credit account balance Change.

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[47]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[48]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# # 5. High Debit Account Balance Change

# In[49]:


training= finalDF[['acc_balance', 'acc_balance_change_amount', 'account_growth', 'cust_tenure', 'channel','age','gender', 'High_limit', 'High_tenure', 'High_bal_change','High_credit_change', 'High_debit_change','churn']].copy()


# In[50]:


from dowhy import CausalModel
from IPython.display import Image, display
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['High_debit_change'],
        outcome='churn')
model.view_model()


# Above are our Causal model and all the assumptions we already think about. 

# #### Identify the Causal Effect

# In[51]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[52]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[53]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~-0.128, where it is equivalent to saying that the probability of churn is increased by ~13% when the customer has a higher debit account balance Change.

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[54]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[55]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# - ## sg_recency ->	days since last day of SG contribution
# - ## promotional_pref ->	customer communication preference
# - ## acc_balance_change_ratio ->	change ratio of account balance
# - ## account_growth ->	precentage of growth assets
# - ## annualrpt_pref ->	annual report preference	
# - ## stmt_pref ->	statement preference

# In[56]:


finalDF['days_sg_recency'] = finalDF['sg_recency'].apply(lambda x: True if x > 10 else False)
finalDF['no_promotional_pref'] = finalDF['promotional_pref'].apply(lambda x: True if x == 'N' else False)
finalDF['positive_acc_balance_change_ratio'] = finalDF['acc_balance_change_ratio'].apply(lambda x: True if x > 0 else False)
finalDF['negative_acc_balance_change_ratio'] = finalDF['acc_balance_change_ratio'].apply(lambda x: True if x < 0 else False)
finalDF['high_account_growth'] = finalDF['account_growth'].apply(lambda x: True if x > 50 else False)
finalDF['low_account_growth'] = finalDF['account_growth'].apply(lambda x: True if x < 20 else False)
finalDF['no_annualrpt_pref'] = finalDF['annualrpt_pref'].apply(lambda x: True if x == 'N' else False)
finalDF['no_stmt_pref'] = finalDF['stmt_pref'].apply(lambda x: True if x == 'N' else False)


# In[62]:


#Creating the 
causal_graph = """
digraph {
age;

gender;
High_credit_change;
High_debit_change;
High_limit;
High_tenure;

cust_tenure;
acc_balance;
days_sg_recency;
no_promotional_pref;
positive_acc_balance_change_ratio;
negative_acc_balance_change_ratio;
high_account_growth;
low_account_growth;
no_annualrpt_pref;
no_stmt_pref;
churn;
acc_balance_change_amount;
age -> acc_balance;
cust_tenure -> acc_balance;
gender -> acc_balance;
age -> acc_balance_change_amount;
age -> High_tenure;
High_limit->churn;
cust_tenure -> High_tenure;
High_limit -> High_credit_change
acc_balance -> High_credit_change

High_limit -> High_debit_change
acc_balance -> High_debit_change
High_bal_change -> churn;
cust_tenure -> churn;
acc_balance -> churn;
days_sg_recency -> churn;
no_promotional_pref -> churn;
positive_acc_balance_change_ratio -> churn;
negative_acc_balance_change_ratio -> churn;
high_account_growth -> churn;
low_account_growth -> churn;
no_annualrpt_pref -> churn;
no_stmt_pref -> churn;
High_credit_change -> churn;
High_debit_change -> churn;
acc_balance_change_amount -> churn;
}
"""


# In[63]:


training= finalDF[['acc_balance', 'acc_balance_change_amount', 'account_growth', 'cust_tenure', 'channel','age','gender', 'High_limit', 'High_tenure', 'High_bal_change','High_credit_change', 'High_debit_change', 
                   'days_sg_recency', 'no_promotional_pref', 'positive_acc_balance_change_ratio', 'negative_acc_balance_change_ratio', 
                   'high_account_growth', 'low_account_growth', 'no_annualrpt_pref','no_stmt_pref', 'churn']].copy()


# # 6. sg_recency -> days since last day of SG contribution
# If greater than 10

# In[64]:


from dowhy import CausalModel
from IPython.display import Image, display
plt.figure(figsize = (17,7))
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['days_sg_recency'],
        outcome='churn')
model.view_model()
plt.show();


# Above are our Causal model and all the assumptions we already think about. 

# #### Identify the Causal Effect

# In[65]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[66]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[67]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~0.15, where it is equivalent to saying that the probability of churn is increased by ~15% when the customer has higher days since last day of SG contribution.

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[68]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[69]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# ## 7. promotional_pref -> customer communication preference

# In[70]:


from dowhy import CausalModel
from IPython.display import Image, display
plt.figure(figsize = (17,7))
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['no_promotional_pref'],
        outcome='churn')
model.view_model()
plt.show();


# Above are our Causal model and all the assumptions we already think about. 

# #### Identify the Causal Effect

# In[71]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[72]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[73]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~-0.0864, where it is equivalent to saying that the probability of churn is decreased by ~8% when the customer has chosen for no promotional perferences.

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[74]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[75]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# ## 8. Positive acc_balance_change_ratio -> change ratio of account balance
# If greater than 0

# In[78]:


from dowhy import CausalModel
from IPython.display import Image, display
plt.figure(figsize = (17,7))
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['positive_acc_balance_change_ratio'],
        outcome='churn')
model.view_model()
plt.show();


# Above are our Causal model and all the assumptions we already think about. 

# #### Identify the Causal Effect

# In[79]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[80]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[81]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~0.15, where it is equivalent to saying that the probability of churn is increased by ~3% when the customer has higher days since last day of SG contribution.

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[82]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[83]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# ## low account_growth -> precentage of growth assets
# if less than 0

# In[85]:


from dowhy import CausalModel
from IPython.display import Image, display
plt.figure(figsize = (17,7))
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['negative_acc_balance_change_ratio'],
        outcome='churn')
model.view_model()
plt.show();


# Above are our Causal model and all the assumptions we already think about. 

# #### Identify the Causal Effect

# In[86]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[87]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[88]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~0.03, where it is equivalent to saying that the probability of churn is increased by ~3% when the customer has negative account growth rate and decreased by 3 % when the customer has positive account growth rate.

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[89]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[90]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# ## 9. annualrpt_pref -> annual report preference

# In[91]:


from dowhy import CausalModel
from IPython.display import Image, display
plt.figure(figsize = (17,7))
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['no_annualrpt_pref'],
        outcome='churn')
model.view_model()
plt.show();


# Above are our Causal model and all the assumptions we already think about. 

# #### Identify the Causal Effect

# In[92]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[93]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[94]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~0.14, where it is equivalent to saying that the probability of churn is increased by ~14% when the customer has not preferred for Annual Performance Report.

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[95]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[96]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# ## 10. stmt_pref -> statement preference

# In[97]:


from dowhy import CausalModel
from IPython.display import Image, display
plt.figure(figsize = (17,7))
model= CausalModel(
        data = training,
        graph=causal_graph.replace("\n", " "),
        treatment=['no_stmt_pref'],
        outcome='churn')
model.view_model()
plt.show();


# Above are our Causal model and all the assumptions we already think about. 

# #### Identify the Causal Effect

# In[98]:


# Identify the causal effect
estimands = model.identify_effect()
print(estimands)


# In[99]:


#Causal Effect Estimation
estimate = model.estimate_effect(estimands,method_name = "backdoor.propensity_score_weighting")
print(estimate)


# In[100]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# ## From the result above, we obtain the mean estimate is ~-0.14, where it is equivalent to saying that the probability of churn is decreased by ~14% when the customer has not preferred for Annual Performance Report.

# Random Common Cause — add an independent random variable as a common cause to the dataset; If the assumption was correct, the estimation should not change.

# In[101]:


refutel = model.refute_estimate(estimands,estimate, "random_common_cause")
print(refutel)


# Data Subset Refuter — replace the given dataset with a randomly selected subset; If the assumption was correct, the estimation should not change that much.

# In[102]:


refutel = model.refute_estimate(estimands,estimate, "data_subset_refuter")
print(refutel)


# In[ ]:





# In[ ]:




