#!/usr/bin/env python
# coding: utf-8

# # Churn Prediction

# Using Deep Learning 
# ---

# Author- David H.

# ####  Introduction
# 
# We aim to accomplist the following for this study:
# 
# 1. Identify and visualize which factors contribute to customer churn:
# 
# 2. Build a prediction model that will perform the following:
# 
# 3. Classify if a customer is going to churn or not
# 
# Preferably and based on model performance, choose a model that will attach a probability to the churn to make it easier for customer service to target low hanging fruits in their efforts to prevent churn

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


def summarize_categoricals(df, show_levels=False):
    """
        Display uniqueness in each column
    """
    data = [[df[c].unique(), len(df[c].unique()), df[c].isnull().sum()] for c in df.columns]
    df_temp = pd.DataFrame(data, index=df.columns,
                           columns=['Levels', 'No. of Levels', 'No. of Missing Values'])
    return df_temp.iloc[:, 0 if show_levels else 1:]


def find_categorical(df, cutoff=10):
    """
        Function to find categorical columns in the dataframe.
    """
    cat_cols = []
    for col in df.columns:
        if len(df[col].unique()) <= cutoff:
            cat_cols.append(col)
    return cat_cols


def to_categorical(columns, df):
    """
        Converts the columns passed in `columns` to categorical datatype
    """
    for col in columns:
        df[col] = df[col].astype('category')
    return df


# In[15]:


finalDF.drop(['new_id'], axis = 1, inplace = True)
finalDF.drop(['postcode'], axis = 1, inplace = True)
finalDF.drop(['group_no'], axis = 1, inplace = True)


# In[16]:


finalDF['churn'] = finalDF['churn'].astype(object)
finalDF['age'] = finalDF['age'].astype(float)
finalDF['num_accounts'] = finalDF['num_accounts'].astype(float)
finalDF['acc_tenure'] = finalDF['acc_tenure'].astype(float)
finalDF['acc_balance'] = finalDF['acc_balance'].astype(float)
finalDF['num_options'] = finalDF['num_options'].astype(float)
finalDF['acc_balance_change_amount'] = finalDF['acc_balance_change_amount'].astype(float)
finalDF['num_options_change_amount'] = finalDF['num_options_change_amount'].astype(float)
finalDF['acc_balance_change_ratio'] = finalDF['acc_balance_change_ratio'].astype(float)
finalDF['num_accounts_change_ratio'] = finalDF['num_accounts_change_ratio'].astype(float)
finalDF['num_options_change_ratio'] = finalDF['num_options_change_ratio'].astype(float)
finalDF['fund_performance'] = finalDF['fund_performance'].astype(float)
finalDF['account_growth'] = finalDF['account_growth'].astype(float)
finalDF['account_growth_change'] = finalDF['account_growth_change'].astype(float)
finalDF['cust_tenure'] = finalDF['cust_tenure'].astype(float)
finalDF['dealer_change_recency'] = finalDF['dealer_change_recency'].astype(float)
finalDF['login_freq'] = finalDF['login_freq'].astype(float)
finalDF['login_recency'] = finalDF['login_recency'].astype(float)
finalDF['call_freq'] = finalDF['call_freq'].astype(float)
finalDF['call_recency'] = finalDF['call_recency'].astype(float)
finalDF['outflow_freq'] = finalDF['outflow_freq'].astype(float)
finalDF['outflow_recency'] = finalDF['outflow_recency'].astype(float)
finalDF['outflow_amount'] = finalDF['outflow_amount'].astype(float)
finalDF['outflow_ratio'] = finalDF['outflow_ratio'].astype(float)
finalDF['adviser_revenue_freq'] = finalDF['adviser_revenue_freq'].astype(float)
finalDF['adviser_revenue_amount'] = finalDF['adviser_revenue_amount'].astype(float)
finalDF['sg_amount'] = finalDF['sg_amount'].astype(float)
finalDF['salary_scr_amount'] = finalDF['salary_scr_amount'].astype(float)
finalDF['spouse_contr_amount'] = finalDF['spouse_contr_amount'].astype(float)
finalDF['personal_contr_amount'] = finalDF['personal_contr_amount'].astype(float)
finalDF['rollover_amount'] = finalDF['rollover_amount'].astype(float)
finalDF['contribution_amount'] = finalDF['contribution_amount'].astype(float)                                                  


# In[17]:


categoricals = find_categorical(finalDF, cutoff=12)


# In[18]:


summarize_categoricals(finalDF[categoricals], show_levels=True)


# In[19]:


numericals = list(set(finalDF.columns.tolist()) - set(categoricals)) + list(set(categoricals) - set(finalDF.columns.tolist()))


# In[20]:


labels = 'Non-Churn', 'Churn'
sizes = [finalDF.churn[finalDF['churn']==0].count(), finalDF.churn[finalDF['churn']==1].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show();


# In[21]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
finalDF.head()


# In[22]:


# Relations based on the categorical data attributes
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='gender', hue = 'churn',data = finalDF, ax=axarr[0][0])
sns.countplot(x='channel', hue = 'churn',data = finalDF, ax=axarr[0][1])
sns.countplot(x='has_mobile', hue = 'churn',data = finalDF, ax=axarr[1][0])
axarr[0][1].set_xticklabels(df['channel'].unique().tolist(), rotation=40)
sns.countplot(x='has_email', hue = 'churn',data = finalDF, ax=axarr[1][1])
plt.tight_layout()
plt.show();


# In[23]:


# Relations based on the continuous data attributes
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.boxplot(y='age',x = 'churn', hue = 'churn',data = finalDF, ax=axarr[0][0])
sns.boxplot(y='acc_tenure',x = 'churn', hue = 'churn',data = finalDF , ax=axarr[0][1])
sns.boxplot(y='acc_balance',x = 'churn', hue = 'churn',data = finalDF, ax=axarr[1][0])
sns.boxplot(y='num_options',x = 'churn', hue = 'churn',data = finalDF, ax=axarr[1][1])
plt.show();


# In[24]:


var = finalDF.select_dtypes(include=['float64', 'int']).columns.tolist()

i = 0
t0 = finalDF.loc[finalDF['churn'] == 0]
t1 = finalDF.loc[finalDF['churn'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,20))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.show();


# ### Correlation between Quantitative variables

# In[25]:


numericals = list(set(finalDF.columns.tolist()) - set(categoricals)) + list(set(categoricals) - set(finalDF.columns.tolist()))


# In[26]:


plt.figure(figsize = (17,14))
plt.title('Correlation plot (Pearson)')
corr = finalDF[numericals].corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show();


# ### There is high correlation among some features. 

# In[27]:


def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr.style.background_gradient(axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# In[28]:


def remove_collinear_features(x, threshold = 0.99):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)

    return x


# In[29]:


finalDF = remove_collinear_features(finalDF, threshold = 0.9)


# In[30]:


print('Dimensions of the Training set:',finalDF.shape)


# In[31]:


final_data = finalDF.copy()


# #### Pre-processing

# In[32]:


# One-hot encoding
def one_hot_encoding(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each, 1)
    return df

# Function to min-max normalize
def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy() # do not touch the original df
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[33]:


categoricals.remove('insurance_amount')
categoricals.remove('insurance_types')
categoricals.remove('churn')


# In[34]:


#categoricals.remove('channel')


# In[35]:


#categoricals.remove('num_accounts_change_ratio')


# In[36]:


# get the intial set of encoded features and encode them
finalDF = one_hot_encoding(finalDF, categoricals)


# In[37]:


print('Dimensions of the Training set:',finalDF.shape)


# In[38]:


finalDF.head()


# ### Splitting into features and labels dataframe

# In[39]:


finalDF = finalDF.dropna()
finalDF.reset_index(inplace = True, drop = True)


# In[40]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_cols = list(finalDF.dtypes[finalDF.dtypes != 'object'].index)
finalDF.loc[:,numeric_cols] = scaler.fit_transform(finalDF.loc[:,numeric_cols])


# In[41]:


finalDF.head()


# In[42]:


finalDF['churn'] = finalDF['churn'].astype('object')
x = finalDF.drop(['churn'], axis = 1)
y = finalDF['churn']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)


# In[43]:


from sklearn.model_selection import train_test_split

data_splits = train_test_split(x, y, test_size=0.2, random_state=0,
                               shuffle=True)
x_train, x_test, y_train, y_test = data_splits


# In[44]:


print('Shape of Training Features : ', x_train.shape)
print('Shape of Testing Features : ', x_test.shape)

print('Shape of Training Labels : ', y_train.shape)
print('Shape of Testing Labels : ', y_test.shape)


# ### Split into train and test

# In[45]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
print('Train data shape: {}'.format(X_train.shape))
print('Test data shape: {}'.format(X_test.shape))


# ## Our Modelling Approach

# <img src = "https://scikit-learn.org/stable/_images/grid_search_workflow.png">

# ## Model Evaluation  <a id='model-eval'></a>
# 
#  In this step we will first define which evaluation metrics we will use to evaluate our model. The most important evaluation metric for this problem domain is **sensitivity, specificity, Precision, F1-measure, Geometric mean and mathew correlation coefficient and finally ROC AUC curve**
#  
#  
# A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. The confusion matrix itself is relatively simple to understand, but the related terminology can be confusing.
# 
# Example confusion matrix for a binary classifier
# 
# <img src = 'https://www.dataschool.io/content/images/2015/01/confusion_matrix_simple2.png'>
# 
# What can we learn from this matrix?
# 
# - There are two possible predicted classes: "yes" and "no". If we were predicting the presence of a disease, for example, "yes" would mean they have the disease, and "no" would mean they don't have the disease.
# - The classifier made a total of 165 predictions (e.g., 165 patients were being tested for the presence of that disease).
# - Out of those 165 cases, the classifier predicted "yes" 110 times, and "no" 55 times.
# - In reality, 105 patients in the sample have the disease, and 60 patients do not.
# 
# Let's now define the most basic terms, which are whole numbers (not rates):
# 
# **true positives (TP):** These are cases in which we predicted yes (they have the disease), and they do have the disease.
# 
# **true negatives (TN):** We predicted no, and they don't have the disease.
# 
# **false positives (FP):** We predicted yes, but they don't actually have the disease. (Also known as a "Type I error.")
# 
# **false negatives (FN):** We predicted no, but they actually do have the disease. (Also known as a "Type II error.")
# 
# I've added these terms to the confusion matrix, and also added the row and column totals:
# 
# 
# This is a list of rates that are often computed from a confusion matrix for a binary classifier:
# 
# **Accuracy:** Overall, how often is the classifier correct?
# (TP+TN)/total = (100+50)/165 = 0.91
# 
# **Misclassification Rate:** Overall, how often is it wrong?
# (FP+FN)/total = (10+5)/165 = 0.09
# equivalent to 1 minus Accuracy
# also known as "Error Rate"
# 
# **True Positive Rate:** When it's actually yes, how often does it predict yes?
# TP/actual yes = 100/105 = 0.95
# also known as "Sensitivity" or "Recall"
# 
# **False Positive Rate:** When it's actually no, how often does it predict yes?
# FP/actual no = 10/60 = 0.17
# 
# **True Negative Rate:** When it's actually no, how often does it predict no?
# TN/actual no = 50/60 = 0.83
# equivalent to 1 minus False Positive Rate
# also known as "Specificity"
# 
# **Precision:** When it predicts yes, how often is it correct?
# TP/predicted yes = 100/110 = 0.91
# 
# **Prevalence:** How often does the yes condition actually occur in our sample?
# actual yes/total = 105/165 = 0.64
# A couple other terms are also worth mentioning:
# 
# **Null Error Rate:** This is how often you would be wrong if you always predicted the majority class. (In our example, the null error rate would be 60/165=0.36 because if you always predicted yes, you would only be wrong for the 60 "no" cases.) 
# 
# This can be a useful baseline metric to compare your classifier against. However, the best classifier for a particular application will sometimes have a higher error rate than the null error rate, as demonstrated by the Accuracy Paradox.
# 
# 
# **Cohen's Kappa:** This is essentially a measure of how well the classifier performed as compared to how well it would have performed simply by chance. In other words, a model will have a high Kappa score if there is a big difference between the accuracy and the null error rate. (More details about Cohen's Kappa.)
# 
# **F Score:** This is a weighted average of the true positive rate (recall) and precision. (More details about the F Score.)
# 
# **ROC Curve:** This is a commonly used graph that summarizes the performance of a classifier over all possible thresholds. It is generated by plotting the True Positive Rate (y-axis) against the False Positive Rate (x-axis) as you vary the threshold for assigning observations to a given class. (More details about ROC Curves.)
# 
# 
# 
# ### Sensitivity vs Specificity

# ![](https://i.ibb.co/d43FVfJ/Sensitivity-and-specificity-svg.png)
# 
# ### Mathew Correlation coefficient (MCC)
# 
# The Matthews correlation coefficient (MCC), instead, is a more reliable statistical rate which produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (true positives, false negatives, true negatives, and false positives), proportionally both to the size of positive elements and the size of negative elements in the dataset.
# 
# ![](https://i.ibb.co/mH6MmG4/mcc.jpg)

# ### Log Loss
# Logarithmic loss  measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value. A perfect model would have a log loss of 0. Log loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high log loss.
# 
# The graph below shows the range of possible log loss values given a true observation (isDog = 1). As the predicted probability approaches 1, log loss slowly decreases. As the predicted probability decreases, however, the log loss increases rapidly. Log loss penalizes both types of errors, but especially those predications that are confident and wrong!
# 
# ![](https://i.ibb.co/6BdDczW/log-loss.jpg)

# ### F1 Score
# 
#  F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. In our case, F1 score is 0.701.
# 
# **F1 Score = 2*(Recall * Precision) / (Recall + Precision)**
# 
# 
# <img src = 'https://www.researchgate.net/publication/325567208/figure/tbl4/AS:668664739151911@1536433505975/Classification-performance-metrics-based-on-the-confusion-matrix.png'>

# # Comparison Table of ML based Models - Train Test Split

# In[46]:


from sklearn.metrics import cohen_kappa_score
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, shuffle=True)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train = X_train.fillna(method='ffill')
X_test = X_test.fillna(method='ffill')


# ### Applying Undersampling on training data

# In[47]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
Y_train = y_train.copy()


# In[48]:


print('Train data shape: {}'.format(X_train.shape))
print('Test data shape: {}'.format(X_test.shape))


# In[49]:


y_train.value_counts()


# ## Deep Learning based Ensembles

# In[50]:


import scipy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam # I believe this is better optimizer for our case
from tensorflow.keras.preprocessing.image import ImageDataGenerator # to augmenting our images for increasing accuracy
from tensorflow.keras.utils import plot_model


# In[51]:


scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# In[52]:


X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)


# In[53]:


def model_ANN1(input_shape=X_train.shape[1], num_classes=2):   
    model = Sequential()

    model.add(Dense(256, activation='tanh', input_dim=X_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    # Lets add softmax activated neurons as much as number of classes
    model.add(Dense(1, activation = "sigmoid"))
    # Compile the model with loss and metrics
    model.compile(optimizer =  Adam(learning_rate = 0.00001, decay = 1e-5) , loss = "binary_crossentropy", metrics=["accuracy"])
    
    return model


# In[54]:


def model_ANN2(input_shape=X_train.shape[1], num_classes=2):   
    model = Sequential()
    model.add(Dense(256, activation='tanh',  input_dim=X_train.shape[1]))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.4))

    # Lets add softmax activated neurons as much as number of classes
    model.add(Dense(1, activation = "sigmoid"))
    # Compile the model with loss and metrics
    model.compile(optimizer =  Adam(learning_rate = 0.00001, decay = 1e-5) , loss = "binary_crossentropy", metrics=["accuracy"])
    
    return model


# In[55]:


ANN_model1 = model_ANN1(input_shape=X_train.shape[1], num_classes=2)
ANN_model2 = model_ANN2(input_shape=X_train.shape[1], num_classes=2)


# In[56]:


ANN_model1.summary()


# In[58]:


ANN_model2.summary()


# In[59]:


dot_img_file = 'model_1.png'
plot_model(ANN_model1, to_file=dot_img_file, show_shapes=True)


# In[60]:


dot_img_file = 'model_2.png'
plot_model(ANN_model2, to_file=dot_img_file, show_shapes=True)


# In[61]:


model = []
model.append(ANN_model1)
model.append(ANN_model2)


# In[62]:


# Start multiple model training with the batch size
# Use Reduce LR on Plateau for reducing Learning Rate if there is no decrease at loss for 3 epochs
models = []
for i in range(len(model)):
    model[i].fit(X_train,y_train, batch_size=512,
                                        epochs = 100,
                                        validation_data = (X_test,y_test), 
                                        callbacks=[ReduceLROnPlateau(monitor='loss', patience=3, factor=0.1)], 
                                        verbose=2)
    models.append(model[i])


# In[63]:


ANN1_preds = (model[0].predict(X_test) > 0.5).astype("int32")
ANN2_preds = (model[1].predict(X_test) > 0.5).astype("int32")
preds = pd.DataFrame({"ANN1" : ANN1_preds.ravel(), "ANN2" : ANN2_preds.ravel()})
ANN_ensemble_predicted = preds.mode(axis=1)
final_preds = ANN_ensemble_predicted.iloc[:, 0].astype("int32")

from sklearn import metrics
CM=metrics.confusion_matrix(y_test,final_preds)
sns.heatmap(CM,annot=True,fmt='2.0f')

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
specificity = TN/(TN+FP)
loss_log = metrics.log_loss(y_test, final_preds)
acc= accuracy_score(y_test, final_preds)
roc=metrics.roc_auc_score(y_test, final_preds)
prec = metrics.precision_score(y_test, final_preds)
rec = metrics.recall_score(y_test, final_preds)
f1 = metrics.f1_score(y_test, final_preds)

mathew = metrics.matthews_corrcoef(y_test, final_preds)    
ck = cohen_kappa_score(y_test, final_preds)

ANN1_preds = (model[0].predict(X_train) > 0.5).astype("int32")
ANN2_preds = (model[1].predict(X_train) > 0.5).astype("int32")
preds = pd.DataFrame({"ANN1" : ANN1_preds.ravel(), "ANN2" : ANN2_preds.ravel()})
ANN_ensemble_predicted = preds.mode(axis=1)
final_preds = ANN_ensemble_predicted.iloc[:, 0].astype("int32")



train_acc= accuracy_score(y_train, final_preds)



# In[64]:


model_results =pd.DataFrame([['ANN Ensemble Classifier',train_acc, acc, prec,rec, f1,roc,mathew, ck]],
               columns = ['Model', 'Train Accuracy', 'Test Accuracy','Precision', 'Recall', 'F1 Score','AUC Score','Matthew Correlation Coefficient', 'Cohen Kappa Score'])


model_results = model_results.set_index('Model')

model_results.index.name = None

model_results


# # Hyper-Parameter Tuning
# 

# In[57]:


from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch


# In[58]:


import tensorflow as tf
from kerastuner import HyperModel

class FCNModel(HyperModel):
    
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(
                units=hp.Choice(
                    'num_filters_1',
                    values=[128, 256, 512],
                    default=256,
                ),
                activation=hp.Choice(
                    'dense_activation1',
                    values=['tanh'],
                    default='tanh'
                ),
                input_dim=self.input_shape
            )
        )
        model.add(
            tf.keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_1',
                    min_value=0.0,
                    max_value=0.6,
                    default=0.2,
                    step=0.1
                )
            )
        )
        model.add(
            tf.keras.layers.Dense(
                units=hp.Choice(
                    'num_filters_2',
                    values=[256, 128, 64, 32],
                    default=64,
                ),
                activation=hp.Choice(
                    'dense_activation2',
                    values=['relu', 'tanh'],
                    default='relu'
                )           )
        )
        model.add(
            tf.keras.layers.Dropout(
                rate=hp.Float(
                    'dropout_2',
                    min_value=0.0,
                    max_value=0.6,
                    default=0.2,
                    step=0.1
                )
            )
        )
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    'num_filters_3',
                    min_value=32,
                    max_value=64,
                    step=4,
                    default=32
                ),
                activation=hp.Choice(
                    'dense_activation3',
                    values=['relu', 'tanh'],
                    default='relu'
                )
            )
        )
        
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    'num_filters_4',
                    min_value=8,
                    max_value=16,
                    step=4,
                    default=8
                ),
                activation=hp.Choice(
                    'dense_activation3',
                    values=['relu', 'tanh'],
                    default='relu'
                )
            )
        )
        model.add(tf.keras.layers.Dense(1, activation='relu'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-6,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

hypermodel = FCNModel(input_shape=X_train.shape[1])


# In[59]:


from kerastuner.tuners import RandomSearch

tuner = RandomSearch(
    hypermodel,
    objective='accuracy',
    seed=123,
    max_trials=100,
    directory='FCN_results',
    project_name='FCN'
)


# In[60]:


tuner.search_space_summary()


# In[61]:


num_of_epochs = 100
tuner.search(X_train, y_train,
             epochs=num_of_epochs,
             validation_data=(X_test, y_test))


# In[62]:


models = tuner.get_best_models(num_models=2)


# In[63]:


import os
import json

vis_data = []
rootdir = 'FCN_results/FCN/'
for subdirs, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith("trial.json"):
            with open(subdirs + '/' + file, 'r') as json_file:
                data = json_file.read()
                vis_data.append(json.loads(data))


# In[64]:


vis_data


# In[68]:


import hiplot as hip
data = [{'num_filters_1': vis_data[idx]['hyperparameters']['values']['num_filters_1'],
         'dense_activation1': vis_data[idx]['hyperparameters']['values']['dense_activation1'], 
         'dropout1': vis_data[idx]['hyperparameters']['values']['dropout_1'], 
         
         'num_filters_2': vis_data[idx]['hyperparameters']['values']['num_filters_2'],
         'dense_activation2': vis_data[idx]['hyperparameters']['values']['dense_activation2'], 
         'dropout2': vis_data[idx]['hyperparameters']['values']['dropout_2'], 
         
         'num_filters_3': vis_data[idx]['hyperparameters']['values']['num_filters_3'],
         'dense_activation3': vis_data[idx]['hyperparameters']['values']['dense_activation3'], 
                  
         'num_filters_4': vis_data[idx]['hyperparameters']['values']['num_filters_4'], 
         
         'learning_rate': vis_data[idx]['hyperparameters']['values']['learning_rate'], 
         'loss': vis_data[idx]['metrics']['metrics']['loss']['observations'][0]['value'],  
         'val_loss': vis_data[idx]['metrics']['metrics']['val_loss']['observations'][0]['value'], 
         'accuracy': vis_data[idx]['metrics']['metrics']['accuracy']['observations'][0]['value'],
         'val_accuracy': vis_data[idx]['metrics']['metrics']['val_accuracy']['observations'][0]['value']} for idx in range(100)]

hip.Experiment.from_iterable(data).display()


# In[69]:


pd.DataFrame(vis_data)


# In[74]:


fcn_results = pd.read_csv('hiplot-selected-100.csv')
fcn_results.set_index('uid', inplace = True)
fcn_results['val_accuracy'].sort_values(ascending = False).plot(kind = 'bar', figsize = (17,9));
plt.title('Validation Accuracy for Deep Learning Ensemble Model');
plt.show();


# In[75]:


fcn_results['val_accuracy'].sort_values(ascending = True)


# Model no. 65 (u_id) is the best model, so the best parameters are the following.

# - num_filters_1 = 128
# - num_filters_2 = 32
# - units = 288
# - dense_activation = tanh
# - learning_rate= 0.000668

# In[87]:


fcn_results.iloc[65, :]


# In[89]:


# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train,
             epochs=100,
             validation_data=(X_test, y_test))

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


# In[94]:


plt.figure(figsize = (17,7))
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy versus Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show();


# ## Tuning Learning Rate

# In[130]:


lr = [0.0001,0.0002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]

def build_model(lr):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
                units=128,
                activation='tanh',
                input_dim=X_train.shape[1]))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(
                units=32,
                activation='relu'))
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Dense(
                units=52,
                activation='relu'))        
    model.add(tf.keras.layers.Dense(
          units=8,
          activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))

    model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    return model


# In[131]:


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model=KerasClassifier(build_fn=build_model)


# In[132]:


from sklearn.model_selection import GridSearchCV
param_grid = dict(lr=lr)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose = 1, scoring = 'accuracy')
grid_result = grid.fit(X_train, Y_train, epochs = 100)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[135]:


lrDF = pd.concat([pd.DataFrame(grid_result.cv_results_["params"]),pd.DataFrame(grid_result.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)


# In[136]:


lrDF


# In[155]:


lrDF.set_index('lr').plot(kind = 'line', figsize = (17,7))
plt.title('Learning Rate versus Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.show();


# ## Tuning Momentum

# In[143]:


momentum = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,  0.6, 0.7,  0.8, 0.9]

def build_model(momentum):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
                units=128,
                activation='tanh',
                input_dim=X_train.shape[1]))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(
                units=32,
                activation='relu'))
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Dense(
                units=52,
                activation='relu'))        
    model.add(tf.keras.layers.Dense(
          units=8,
          activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))

    model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.00010, momentum=momentum),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    return model


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model=KerasClassifier(build_fn=build_model)


# In[144]:


param_grid = dict(momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose = 1, scoring = 'accuracy')
grid_result = grid.fit(X_train, Y_train, epochs = 100)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
momentumDF = pd.concat([pd.DataFrame(grid_result.cv_results_["params"]),pd.DataFrame(grid_result.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)    


# In[151]:


momentumDF


# In[156]:


momentumDF.set_index('momentum').plot(kind = 'line', figsize = (17,7))
plt.title('Momentum versus Accuracy')
plt.xlabel('Momentum')
plt.ylabel('Accuracy')
plt.show();


# ## Tune Network Weight Initialization 

# In[145]:


init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

def build_model(init_mode):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
                units=128,
                activation='tanh',kernel_initializer=init_mode,
                input_dim=X_train.shape[1]))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(
                units=32,
                activation='relu'))
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Dense(
                units=52,
                activation='relu'))        
    model.add(tf.keras.layers.Dense(
          units=8,
          activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))

    model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.00010, momentum=0.1),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    return model


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model=KerasClassifier(build_fn=build_model)


# In[146]:


param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose = 1, scoring = 'accuracy')
grid_result = grid.fit(X_train, Y_train, epochs = 100)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
initialisersDF = pd.concat([pd.DataFrame(grid_result.cv_results_["params"]),pd.DataFrame(grid_result.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)    


# In[150]:


initialisersDF


# In[157]:


initialisersDF.set_index('init_mode').plot(kind = 'line', figsize = (17,7))
plt.title('Kernel Initializers versus Accuracy')
plt.xlabel('Kernel Initializers')
plt.ylabel('Accuracy')
plt.show();


# ## Tune Dropout Regularization

# In[147]:


dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def build_model(dropout_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
                units=128,
                activation='tanh',kernel_initializer='glorot_uniform',
                input_dim=X_train.shape[1]))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(
                units=32,
                activation='relu'))
    model.add(tf.keras.layers.Dropout(0.0))
    model.add(tf.keras.layers.Dense(
                units=52,
                activation='relu'))        
    model.add(tf.keras.layers.Dense(
          units=8,
          activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))

    model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.00010, momentum=0.1),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    return model


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model=KerasClassifier(build_fn=build_model)


# In[148]:


param_grid = dict(dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose = 1, scoring = 'accuracy')
grid_result = grid.fit(X_train, Y_train, epochs = 100)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
dropout_rateDF = pd.concat([pd.DataFrame(grid_result.cv_results_["params"]),pd.DataFrame(grid_result.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)    


# In[149]:


dropout_rateDF


# In[158]:


dropout_rateDF.set_index('dropout_rate').plot(kind = 'line', figsize = (17,7))
plt.title('Dropout Rate versus Accuracy')
plt.xlabel('Dropout Rate')
plt.ylabel('Accuracy')
plt.show();


# In[ ]:




