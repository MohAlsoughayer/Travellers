# Author: Mohammed Alsoughayer
# Description: Perform an ML regression model to predict the price of flights based on certain features
# %%
# Helper packages
import numpy as np
import pandas as pd
import math
from plotnine import ggplot, aes, geom_density, geom_line, geom_point, ggtitle
import plotly.express as px

# Modeling process
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, roc_auc_score
# %%
df = pd.read_csv('../data/final_data.csv')
# create train/test split
train, test = train_test_split(df, train_size=0.7)
# get predictors and response varaibles for each set 
X_train = train.drop(["Price","year"], axis=1)
y_train = train[["Price"]]
X_test = test.drop(["Price","year"], axis=1)
y_test = test[["Price"]]
# %%
# Multiple linear regression with kfold repeat (10 folds, 5 repeats)
# define loss function
lossFn = 'neg_root_mean_squared_error'

# create 10 fold CV object
rfk = RepeatedKFold(n_splits=10, n_repeats=5)

# create LM model object
lm_mod = linear_model.LinearRegression()

# execute and score the cross validation procedure
results = cross_val_score(
    estimator=lm_mod,
    X=X_train,
    y=y_train,
    cv=rfk,
    scoring=lossFn
)
results.mean()
# %%
