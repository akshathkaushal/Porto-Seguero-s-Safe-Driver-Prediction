#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# change eda=n if you don't want to do eda
eda = 'n'


# In[1]:

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Team: Y So Serious')
parser.add_argument('--train_set', type=str, nargs=1,  help='path for train dataset')
parser.add_argument('--test_set', type=str, nargs=1, help='path for test dataset')
args = parser.parse_args()

train_set_path, test_set_path = args.train_set[0], args.test_set[0]





import numpy as np
#import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
#tqdm.pandas(tqdm_notebook)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


import warnings
warnings.filterwarnings("ignore")

# importing baseline models
from sklearn import linear_model
from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import catboost
import lightgbm as lgbm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from mlxtend.classifier import StackingClassifier


# In[ ]:


# Code for calculating Normalized gini coefficient
# Taken from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703
def gini(actual, pred, cmpcol = 0, sortcol = 1):  
    assert( len(actual) == len(pred) )  
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)  
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]  
    totalLosses = all[:,0].sum()  
    giniSum = all[:,0].cumsum().sum() / totalLosses  
  
    giniSum -= (len(actual) + 1) / 2  
    return giniSum / len(actual)  
  
def gini_normalized(a, p):  
    return gini(a, p) / gini(a, a)

# Impute Missing Values
def impute(df, cols, strat):
    if(strat=="constant"):
        x = input("Enter the string: ")
        imputer= Imputer(strategy=strat, fill_value=str(x))
    else:
        imputer = Imputer(strategy=strat)
    df_impute = df[cols]
    df[cols] = imputer.fit_transform(df_impute)
    return df

## Removing the outliers in the box plots
def remove_outliers(x,df,col):
    q1 = df[col].quantile(q = 0.25)
    q3 = df[col].quantile(q = 0.75)
    iqr = q3 - q1
    outlier_range = 1.5*iqr
    r_whisker = q3 + outlier_range
    l_whisker = q1 - outlier_range
    if (x > r_whisker):
        return q3
    elif (x < l_whisker):
        return q1
    else: 
        return x    
    
def check_missing_data(data):
    d = pd.DataFrame()
    for val in data.columns.values:
        count = data[data[val]==-1].shape[0]
        vals = pd.DataFrame({"Count":count,"%":(count/data[val].shape[0])*100}, index=[val])
        d = d.append(vals)
    return d
        
def one_hot_encode(train, test, cat_cols):
    train = pd.get_dummies(train, columns=cat_cols)
    test = pd.get_dummies(test, columns=cat_cols)
    return train, test

def check_data(df):
    vals=df.isnull().sum()
    percent=(100*vals)/len(df)
    type_data=df.dtypes
    diff_vals = df.nunique()
    sk = df.skew()
    table = pd.concat([vals,percent,type_data,diff_vals,sk],axis=1)
    table = table.rename(columns={0:"missing", 1:"missing %", 2:"dtype", 3:"# unique", 4:"skew"})
    return table

def get_scoring_plots(train_pred, test_pred, train_Y, test_Y):
    f, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    train_cnf = confusion_matrix(train_Y, np.round(train_pred))
    val_cnf = confusion_matrix(test_Y, np.round(test_pred))

    train_cnf = train_cnf / train_cnf.sum(axis=1)[:, np.newaxis]
    val_cnf = val_cnf / val_cnf.sum(axis=1)[:, np.newaxis]

    train_df_cm = pd.DataFrame(train_cnf, index=[0, 1], columns=[0, 1])
    val_df_cm = pd.DataFrame(val_cnf, index=[0, 1], columns=[0, 1])

    sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="inferno", ax=axes[0]).set_title("Train")
    sns.heatmap(val_df_cm, annot=True, fmt='.2f', cmap="inferno", ax=axes[1]).set_title("Validation")
    plt.show()
    
def get_scoring_metrics(y_test, pred):
    f1 = f1_score(y_test,np.round(pred))
    acc = accuracy_score(y_test,np.round(pred))
    rc = recall_score(y_test,np.round(pred))
    prec = precision_score(y_test,np.round(pred))
    
    print("F1 score: ",f1)
    print("Accuracy: ",acc)
    print("Recall:",rc)
    print("Precision: ",prec)


# In[ ]:


# download the datasets
#train = pd.read_csv(r"C:\Users\Administrator\Desktop\ML project data downloaded from kaggle\train.csv")
#test = pd.read_csv(r"C:\Users\Administrator\Desktop\ML project data downloaded from kaggle\test.csv")

train = pd.read_csv(train_set_path)
print(train.head())

test = pd.read_csv(test_set_path)
print(test.head())


# In[ ]:


#check_missing_data(train)


# # Getting to know the dataset

# **Pre existing knowledge about data:**
# 
# 1. target column in train dataset is the output.
# 2. There are no missing values in the data, they have been replaced by -1.
# 3. There are calculated features.

# In[ ]:


#print(train.shape)
#print(test.shape)


# In[ ]:


#train.head()


# In[ ]:


#train.info()


# In[ ]:


#check_data(train)


# In[ ]:


#train.describe()


# **Separating useful columns**

# In[ ]:


train_target = train.target
test_id = test.id
train.drop(columns=['id','target'],inplace=True)
test.drop(columns=['id'],inplace=True)


# **Target Analysis**

# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(15,8))
    f = sns.countplot(train_target)
    for p in f.patches:
        f.annotate('{:.2f}%'.format(100*p.get_height()/len(train_target)), 
                    (p.get_x() + 0.3, p.get_height() + 10000))


# **Analysing and separating the columns**

# In[ ]:


def analyse_columns(df):
    data = []
    for col in df.columns:
        if 'bin' in col:
            col_type = 'binary'
        elif 'cat' in col:
            col_type = 'nominal'
        elif df[col].dtype == np.float64:
            col_type = 'interval'
        elif df[col].dtype == np.int64:
            col_type = 'ordinal'
        
        data_type = df[col].dtype
        
        data_dict = {
            'column':col,
            'type':col_type,
            'data':data_type
        }
        
        data.append(data_dict)
    meta = pd.DataFrame(data,columns=['column', 'type', 'data'])
    return meta


# In[ ]:


columns_analysis = analyse_columns(train)
data_type_count = columns_analysis.groupby(['type']).agg({'data': lambda x: x.count()}).reset_index()
#print(data_type_count)
if(eda=='y'):
    sns.barplot(data=data_type_count, x='type', y='data')


# **Separating the columns**

# In[ ]:


cat_cols = train.loc[:,train.columns.str.contains("cat")]
bin_cols = train.loc[:,train.columns.str.contains("bin")]
num_cols = train.drop(columns=['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat', 'ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_16_bin','ps_ind_17_bin', 'ps_ind_18_bin','ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat','ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat','ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat','ps_calc_15_bin','ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin','ps_calc_20_bin'])


# # Exploratory Data Analysis

# **Binary columns**

# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(bin_cols.columns.nunique(),bin_cols.columns.nunique()))
    sns.heatmap(bin_cols.corr(), cmap="coolwarm",vmin=-1, annot=True, linewidths=0.1)


# Hence, no serious correlation between the binary columns.

# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(20,20))
    plt.title("Count plots of binary features")
    i=1
    for col in bin_cols:
        plt.subplot(5,4,i)
        i=i+1
        sns.countplot(bin_cols[col],hue=train_target,palette="jet_r")
        plt.tight_layout()


# From the plots, it is evident that except some of the calc values, most features are dominated by a single value.

# **Categorical data**

# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(cat_cols.columns.nunique(),cat_cols.columns.nunique()))
    sns.heatmap(cat_cols.corr(), cmap="coolwarm",vmin=-1, annot=True, linewidths=0.1)


# As in binary columns, there is no evident correlation between categorical columns.

# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(20,20))
    plt.title("Count plots of categorical features")
    i=1
    for col in cat_cols:
        plt.subplot(5,4,i)
        i=i+1
        sns.countplot(cat_cols[col],hue=train_target,palette="jet_r")
        plt.tight_layout()


# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(20,10))
    sns.countplot(cat_cols['ps_car_11_cat'],hue=train_target,palette="jet_r")


# 1. Some features like 'ps_ind_05_cat' and 'ps_car_04_cat' mostly consists of a single value. Therefore the mode of these features can be used for filling their missing values
# 2. We plan to replace the -1s with either mean, median or mode in categorical values.
# 3. In ps_car_03_cat and In ps_car_05_cat, there is a higher number of -1s, so we can either drop the column or treat -1 as a separate category.
# 4. After all this, we will go for techniques like OHE and Label encoding etc.

# **Numerical data**

# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(num_cols.columns.nunique(),num_cols.columns.nunique()))
    sns.heatmap(num_cols.corr(), cmap="coolwarm",vmin=-1, annot=True, linewidths=0.1)


# In numerical columns, there is some correlation, though not enough to make a significant impact.

# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(20,20))
    i=1
    for col in num_cols.columns:
        plt.subplot(6,6,i)
        i=i+1
        sns.scatterplot(x=col, y=train_target, hue=train_target, data=num_cols)
        plt.tight_layout()


# We separate the numerical features into two separate datasets, one with calc and other without calc.

# In[ ]:


num_calc_cols = num_cols.loc[:,num_cols.columns.str.contains("calc")]
num_cols_no_calc = num_cols.drop(num_calc_cols.columns, axis=1)
num_cols = num_cols_no_calc


# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(20,20))
    i=1
    for col in num_cols_no_calc:
        plt.subplot(5,3,i)
        i=i+1
        sns.boxplot(x=train_target, y=num_cols_no_calc[col], hue=train_target)
        plt.tight_layout()


# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(20,20))
    i=1
    for col in num_cols_no_calc:
        plt.subplot(5,3,i)
        i=i+1
        sns.distplot(num_cols_no_calc[col], kde=False)
        plt.tight_layout()


# Now, for calc numerical columns

# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(20,20))
    i=1
    for col in num_calc_cols:
        plt.subplot(5,3,i)
        i=i+1
        sns.distplot(num_calc_cols[col])
        plt.tight_layout()


# We see that all the cal features have a normal distribution, which is a good thing. But the symmetry can cause a problem in classification. Also, since they are calculated features, the concerns of their relavance to the model arise. Hence, to confirm our suspicions over usability of calc features, we plot boxplots.

# In[ ]:


if(eda=='y'):
    plt.figure(figsize=(20,20))
    i=1
    for col in num_calc_cols:
        plt.subplot(5,3,i)
        i=i+1
        sns.boxplot(x=train_target, y=num_calc_cols[col], hue=train_target)
        plt.tight_layout()


# As expected, we see that boxplots are more or like identical for both target 0 and 1. Hence, they behave as random noise and donot provide any relavant information to the model. So we choose to drop them.

# 1) LightGBM 

# In[ ]:


#
# lightgbm parameters
'''lgb_params = dict(
    learning_rate=[0.005, 0.01, 0.03],
    n_estimators=[500, 700, 900],
    reg_alpha=[2, 4, 6],
    reg_lambda=[2, 4, 6],
    num_leaves=[10, 20, 30],
    subsample=[0.7, 0.8, 0.9]
)
lg = lgbm.LGBMClassifier()
lg_rscv = RandomizedSearchCV(estimator=lg, param_distributions=lgb_params, verbose=1, cv=3, return_train_score=True, n_jobs=-1)
lg_rscv.fit(train, train_target)
print('Best Score: ', lg_rscv.best_score_)
print('Best Params: ', lg_rscv.best_params_)'''


# In[ ]:


# After a little tweaking, we finally adjusted on the following parameters
params = {
    "objective":'binary',
    "boosting_type":"gbdt",
    "is_unbalance":True,
    "learning_rate":0.01,
    "n_estimators":700,
    "reg_alpha":4,
    "reg_lambda":4,
    "num_leaves":10,
    "subsample":0.8
}
lg = lgbm.LGBMClassifier(**params)

print("Starting LightGBM Kfold!")

i=1
# doing kfold cross validation
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=21)
for train_index, test_index in kf.split(train, train_target):
    x_train, x_test = train.iloc[train_index], train.iloc[test_index]
    y_train, y_test = train_target[train_index], train_target[test_index]
    lg.fit(x_train, y_train)
    ans = lg.predict_proba(x_test)[:,1]
    g = gini_normalized(y_test, ans)
    print("KFold {}: {}".format(i,g))
    i=i+1

print("Kflod for LightGBM done, starting fitting on train data!")

# In[ ]:


# fit on the entire dataset and predict values on test data
lg.fit(train, train_target)
ans_lgb = lg.predict_proba(test)[:,1]

print("LightGBM fitted on the dataset and predictions are made!")

# Feature importances

# In[ ]:


feat_imp = pd.DataFrame({"Feature":train.columns, "Importance":lg.feature_importances_},index=range(1,len(train.columns)+1))
#feat_imp


# In[ ]:

if(eda=='y'):
    plt.figure(figsize=(20,20))
    plt.xticks(rotation=45)
    sns.barplot(x=train.columns, y=lg.feature_importances_)


# In[ ]:


zero_imp = feat_imp[feat_imp["Importance"]==0]
list(zero_imp["Feature"])


# In[ ]:


# drop the features with zero importance
train.drop(columns=list(zero_imp["Feature"]),inplace=True)
test.drop(columns=list(zero_imp["Feature"]),inplace=True)

num_cols.drop(columns=['ps_ind_14'],inplace=True)
bin_cols.drop(columns=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin',
'ps_calc_20_bin'],inplace=True)


# # Pre-processing the data

# Imputing missing values:
# 
# **1) Numerical columns**
# 
# We impute like this ->
# 
#      column      metric
#     ps_reg_03     mean
#     ps_car_11     mode
#     ps_car_14     mean
# 
# 
# **2) Categorical columns**
# 
# We impute with the strategy as mode ->
# 
#        column        metric
#     ps_ind_02_cat     mode
#     ps_ind_04_cat     mode
#     ps_ind_05_cat     mode
#     ps_car_01_cat     mode
#     ps_car_02_cat     mode
#     ps_car_07_cat     mode
#     ps_car_09_cat     mode
# 
#     ps_car_03_cat     TaS*
#     ps_car_05_cat     TaS*
#     *Treated as separate
# 
# **3) Binary columns**
# 
# They don't have any missing values.

# 2) Catboost

# In[ ]:


# catboost parameters
'''parameters = dict(
    n_estimators = [1000, 1500, 2000],
    learning_rate = [0.01, 0.03, 0.05],
    depth = [5, 7, 9]
)

cat = catboost.CatBoostClassifier(verbose=False, task_type='GPU')
cat_rscv = RandomizedSearchCV(estimator=cat, param_distributions=parameters, verbose=1, cv=3, return_train_score=True, n_jobs=-1)

cat_rscv.fit(train, train_target)
print('Best Score: ', cat_rscv.best_score_)
print('Best Params: ', cat_rscv.best_params_)'''

print("Starting catboost fitting.")

# In[ ]:


# we adjusted on these final features
cat = catboost.CatBoostClassifier(n_estimators=1000, learning_rate=0.03, depth=7)
cat.fit(train, train_target)
ans_cat = cat.predict_proba(test)[:,1]

print("Catboost fitted and predictions made!")

# In[ ]:


# the second submission file
test_ans2 = 0.9*ans_cat + 0.1*ans_lgb

# In[ ]:

# second file that we submitted
#submit = pd.DataFrame()
#submit["id"] = test_id
#submit["target"] = test_ans2
#submit.to_csv('YSoSerious_submission2.csv',index=False)
#print("Saved to csv!")

#print("Seconds file that was submitted is made!!")

# **Filling in missing values**

# In[ ]:

print("Starting preprocessing data!")

# Numerical columns
num_cols = impute(num_cols, ['ps_reg_03', 'ps_car_14'], 'mean')
num_cols = impute(num_cols, ['ps_car_11'], 'most_frequent')

# Categorical columns
# whether to do this or not, check again
cat_cols = impute(cat_cols, ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_07_cat', 'ps_car_09_cat'], 'most_frequent')


# In[ ]:


training_data = pd.concat([num_cols,cat_cols,bin_cols],axis=1)

# making sure that the train and test datasets have same columns
#print("Train: ",training_data.columns.nunique())
cls = training_data.columns
testing_data = test[cls]
#print("Test: ",testing_data.columns.nunique())


# Encoding

# In[ ]:


train_ohe, test_ohe = one_hot_encode(training_data, testing_data, cat_cols.columns)


# Upsampling

# In[ ]:


X = pd.concat([train_ohe, train_target], axis=1)
not_claimed = X[X.target==0]
claimed = X[X.target==1]
n_samp = round((len(not_claimed))/2)
fraud_upsampled = resample(claimed, replace=True, n_samples=n_samp, random_state=21)
upsampled = pd.concat([fraud_upsampled, not_claimed])
upsampled.target.value_counts()
train_target = upsampled.target
train_ohe = upsampled.drop(['target'],axis=1)


# In[ ]:


#print(test.shape)
#print(train_ohe.shape)


# In[ ]:


#print("training data columns = ", train_ohe.columns.nunique())
#print("testing data columns = ", test_ohe.columns.nunique())
cols = train_ohe.columns
test_ohe = test_ohe[cols]
#print("training data columns = ", train_ohe.columns.nunique())
#print("testing data columns = ", test_ohe.columns.nunique())


# Scaling the data

# In[ ]:


scaler = StandardScaler()
scaler.fit(train_ohe[num_cols.columns])
train_ohe[num_cols.columns] = scaler.transform(train_ohe[num_cols.columns])
test_ohe[num_cols.columns] = scaler.transform(test_ohe[num_cols.columns])

print("Preprocessing of data done!")

# Splitting the dataset

# In[ ]:


train_X, train_CV, target_X, target_CV = tts(train_ohe, train_target, test_size=0.2, random_state=21)
#print("train_X shape = {}".format(train_X.shape))
#print("train_CV shape = {}".format(train_CV.shape))


# # Training models

# **Here is the list of models we will be training:**
# 
# * Linear regression
# * Logistic regression
# * Decision Tree
# * SVM
# * Naive Bayes
# * Random Forest
# * GBM
# * XGBoost
# * LightGBM
# * Catboost

# 3) Linear Regression

# In[ ]:


'''score = {}
alpha = [i for i in np.arange(0.01, 0.101, 0.01)]
for i in tqdm(alpha):
    ridge = linear_model.Ridge(alpha=i)
    ridge.fit(train_X, target_X)
    g = gini_normalized(ridge.predict(train_CV), target_CV)
    score[i] = g


# In[ ]:


dict(sorted(score.items(), key=lambda item: item[1], reverse=True))


# 4) Logistic regression

# In[ ]:


lr = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]
gini_score = {}
for i in tqdm(lr):
    model = SGDClassifier(loss='log', alpha=i, n_jobs=-1, class_weight='balanced')
    model.fit(train_X, target_X)
    gini_score[i] = gini_normalized(target_CV, model.predict_proba(train_CV)[:,1])


# In[ ]:


sgd_lr = max(gini_score, key=gini_score.get)
print("Highest accuracy = ", gini_score[sgd_lr])
print("Best parameter = ", sgd_lr)


# In[ ]:


final_sgd_model = SGDClassifier(loss='log', alpha=sgd_lr, n_jobs=-1, class_weight='balanced')

final_sgd_model.fit(train_X, target_X)
train_val = final_sgd_model.predict_proba(train_X)[:,1]
test_val = final_sgd_model.predict_proba(train_CV)[:,1]

final_sgd_model.fit(train_ohe, train_target)
ans_sgd = final_sgd_model.predict_proba(test_ohe)[:,1]


# In[ ]:


if(eda=='y'):
    get_scoring_plots(train_val, test_val, target_X, target_CV)


# 5) SVM

# In[ ]:


svm = SVC(max_iter=100, C=0.5, probability=True, class_weight='balanced', random_state=21)
svm.fit(train_X, target_X)

train_val = svm.predict_proba(train_X)[:,1]
test_val = svm.predict_proba(train_CV)[:,1]

g = gini_normalized(target_CV, svm.predict_proba(train_CV)[:,1])
print(g)
svm.fit(train_ohe, train_target)
ans_svm = svm.predict_proba(test_ohe)[:,1]


# In[ ]:


if(eda=='y'):
    get_scoring_plots(train_val, test_val, target_X, target_CV)


# 6) Naive Bayes

# In[ ]:


modelG = GaussianNB(var_smoothing =1e+6)
modelG.fit(train_X, target_X)
modelG.get_params(deep=True)

train_val = modelG.predict_proba(train_X)[:,1]
test_val = modelG.predict_proba(train_CV)[:,1]

scorerg = gini_normalized(target_CV,test_val)
#
#modelNB = MultinomialNB(alpha=2.0,fit_prior=False)
#modelNB.fit(train_X, target_X)
#ansnb = modelNB.predict_proba(train_CV)
#scorernb = gini_normalized(target_CV,ansnb[:,1])
#
#modelCNB = ComplementNB(alpha=3,norm=False,fit_prior=False)
#modelCNB.fit(train_X, target_X)
#anscnb = modelCNB.predict_proba(train_CV)
#scorercnb = gini_normalized(target_CV,anscnb[:,1])
#
#modelbNB = BernoulliNB(alpha=5,binarize=0)
#modelbNB.fit(train_X, target_X)
#ansbnb = modelbNB.predict_proba(train_CV)
#scorerbnb = gini_normalized(target_CV,ansbnb[:,1])
#
#modelcaNB = CategoricalNB(alpha=5)
#modelcaNB.fit(train_X, target_X)
#har = modelcaNB.predict_proba(train_CV)
#scorercanb = gini_normalized(target_CV,har[:,1])

print("Gaussian Naive Bayes ",scorerg)
#print("Multinomial Naive Bayes ",scorernb)
#print("Complement Naive Bayes ",scorercnb)
#print("Bernoulli Naive Bayes ",scorerbnb)
#print("Categorical Naive Bayes ",scorercanb)


# In[ ]:


if(eda=='y'):
    get_scoring_plots(train_val, test_val, target_X, target_CV)


# 7) Decision Tree

# In[ ]:


# parameters for decision tree classifier
parameters = dict(
    criterion = ["gini"],
    splitter = ["best"],
    class_weight = ["balanced"],
    max_features = [23, 25, 27],
    max_depth = [11, 13, 15],
    min_samples_split = [1, 3, 5],
    min_samples_leaf = [1, 3, 5],
    max_leaf_nodes = [2000, 4000, 6000],
    min_impurity_split = [0.05, 0.1, 0.3]
)


# In[ ]:


dt = DecisionTreeClassifier()
dt_rscv = RandomizedSearchCV(estimator=dt, param_distributions=parameters, verbose=1, cv=3, return_train_score=True, n_jobs=-1)

dt_rscv.fit(train_ohe, train_target)
print('Best Score: ', dt_rscv.best_score_)
print('Best Params: ', dt_rscv.best_params_)


# In[ ]:


dsct = DecisionTreeClassifier(**dt_rscv.best_params_)
dsct.fit(train_ohe, train_target)

train_val = dsct.predict_proba(train_X)[:,1]
test_val = dsct.predict_proba(train_CV)[:,1]

ans_dsct = dsct.predict_proba(test_ohe)[:,1]


# In[ ]:


if(eda=='y'):
    get_scoring_plots(train_val, test_val, target_X, target_CV)


# 8) Adaboost

# In[ ]:


ada = AdaBoostClassifier(base_estimator = dsct, n_estimators=1000, learning_rate=0.01, algorithm="SAMME")
ada.fit(train_X, target_X)

train_val = ada.predict_proba(train_X)[:,1]
test_val = ada.predict_proba(train_CV)[:,1]

print(gini_normalized(target_CV, test_val))


# In[ ]:


if(eda=='y'):
    get_scoring_plots(train_val, test_val, target_X, target_CV)


# In[ ]:


ada.fit(train_ohe, train_target)
ans_ada = ada.predict_proba(test_ohe)[:,1]


# 9) GradientBoostingClassifier

# In[ ]:


gbc_params = dict(
    learning_rate = [0.05,0.1],
    n_estimators = [150,200,250]
)
gbc = GradientBoostingClassifier()
gbc_rscv = RandomizedSearchCV(estimator=gbc, param_distributions=gbc_params, verbose=1, cv=2, return_train_score=True, n_jobs=-1)

gbc_rscv.fit(train_ohe, train_target)
print('Best Score: ', gbc_rscv.best_score_)
print('Best Params: ', gbc_rscv.best_params_)


# In[ ]:


# after submitting some predictions, we found these to be the best fitting values
gbc = GradientBoostingClassifier(n_estimators=250, learning_rate=0.1, verbose=True)
gbc.fit(train_X, target_X)

train_val = gbc.predict_proba(train_X)[:,1]
test_val = gbc.predict_proba(train_CV)[:,1]

print(gini_normalized(target_CV, test_val))


# In[ ]:


if(eda=='y'):
    get_scoring_plots(train_val, test_val, target_X, target_CV)


# In[ ]:


gbc.fit(train_ohe, train_target)
ans_gbc = gbc.predict_proba(test_ohe)[:,1]'''


# 10) XGBoost

# In[ ]:


# grid search for xgboost
'''xgb_params = dict(
        n_estimators=[100, 150, 200],
        max_depth=[3, 5, 7],
        learning_rate=[0.05, 0.1],
        min_child_weight=[99, 100, 101],
        max_leaves=[5, 7, 9],
        reg_lamda=[0.5]
)

xg = xgb.XGBClassifier(n_jobs=-1, tree_method='gpu_hist', grow_policy="lossguide", sampling_method="gradient_based", update="prune")
xg_rscv = RandomizedSearchCV(estimator=xg, param_distributions=xgb_params, verbose=1, cv=5, return_train_score=True, n_jobs=-1)

xg_rscv.fit(train_ohe, train_target)
print('Best Score: ', xg_rscv.best_score_)
print('Best Params: ', xg_rscv.best_params_)'''


# In[ ]:


final_xgb_model_params = {
    "n_estimators":150,
    "learning_rate":0.1,
    "max_depth":5,
    "subsample":0.8,
    "min_child_weight":101
}


# In[ ]:

print("Fitting the XGBoost Classifier!")

#
final_xgb_model = xgb.XGBClassifier(**final_xgb_model_params, n_jobs=-1, grow_policy="lossguide", sampling_method="gradient_based", update="prune")
#final_xgb_model.fit(train_X, target_X)

#train_val = final_xgb_model.predict_proba(train_X)[:,1]
#test_val = final_xgb_model.predict_proba(train_CV)[:,1]

#print(gini_normalized(target_CV, test_val))


# In[ ]:


if(eda=='y'):
    get_scoring_plots(train_val, test_val, target_X, target_CV)


# In[ ]:


final_xgb_model.fit(train_ohe, train_target)
ans_xgb = final_xgb_model.predict_proba(test_ohe)[:,1]

print("XGBoost fitted and predictions made!")

# 11) Random Forest Classifier

# In[ ]:


# random search for randomforestclassifier
'''rfc_params = dict(
        n_estimators=[300, 500, 700, 1000],
        max_depth=[3, 5, 9, 11],
)

rfc = RandomForestClassifier()
rfc_rscv = RandomizedSearchCV(estimator=rfc, param_distributions=rfc_params, verbose=1, cv=2, return_train_score=True, n_jobs=-1)

rfc_rscv.fit(train_ohe, train_target)
print('Best Score: ', rfc_rscv.best_score_)
print('Best Params: ', rfc_rscv.best_params_)


# In[ ]:


rfc = RandomForestClassifier(**rfc_rscv.best_params_, criterion="gini", n_jobs=-1, verbose=1, oob_score=True)
rfc.fit(train_X, target_X)

train_val = rfc.predict_proba(train_X)[:,1]
test_val = rfc.predict_proba(train_CV)[:,1]

print(gini_normalized(target_CV, test_val))


# In[ ]:


get_scoring_metrics(target_CV, test_val)


# In[ ]:


if(eda=='y'):
    get_scoring_plots(train_val, test_val, target_X, target_CV)


# In[ ]:


rfc.fit(train_ohe, train_target)
ans_rfc = rfc.predict_proba(test_ohe)[:,1]'''


# # Ensembling

# **1) Combining models**

# In[ ]:


test_ans = 0.8*ans_cat + 0.2*ans_xgb


# In[ ]:


submit = pd.DataFrame()
submit["id"] = test_id
submit["target"] = test_ans
submit.to_csv('submission.csv',index=False)
print("Saved to csv!")

print("First file that was submitted is generated!!")

'''
# **2) Stacking**

# In[ ]:


sclf = StackingClassifier(classifiers=[lg, cat, ada, gbc, rfc, final_sgd_model, dsct], use_probas=True, meta_classifier=final_xgb_model, use_features_in_secondary = True)
stacked_model = sclf.fit(train_X, target_X)

train_val = stacked_model.predict_proba(train_X)[:,1]
test_val = stacked_model.predict_proba(train_CV)[:,1]

print(gini_normalized(target_CV, test_val))


# In[ ]:


if(eda=='y'):
    get_scoring_plots(train_val, test_val, target_X, target_CV)


# In[ ]:


get_scoring_metrics(target_CV, test_val)


# In[ ]:


test_ans = stacked_model.predict_proba(test_ohe)[:,1]'''

