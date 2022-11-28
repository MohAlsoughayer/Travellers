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
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
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
lm_fit = lm_mod.fit(X_train, y_train)

# execute and score the cross validation procedure
results = cross_val_score(
    estimator=lm_mod,
    X=X_train,
    y=y_train,
    cv=rfk,
    scoring=lossFn
)
print(results.mean())
print(lm_fit.coef_)
# %%
# K nearest neighbor
# basic model object
knn = KNeighborsRegressor()

# Create grid of hyperparameter values
hyper_grid = {'n_neighbors': range(2, 26)}

# Tune a knn model using grid search
grid_search = GridSearchCV(knn, hyper_grid, cv=rfk, scoring=lossFn)
results = grid_search.fit(X_train, y_train)

# Best model's cross validated RMSE
print(abs(results.best_score_))

# Best model's k value
print(results.best_estimator_.get_params().get('n_neighbors'))
# Plot all RMSE results
all_rmse = pd.DataFrame({'k': range(2, 26), 
                         'RMSE': np.abs(results.cv_results_['mean_test_score'])})

(ggplot(all_rmse, aes(x='k', y='RMSE'))
 + geom_line()
 + geom_point()
 + ggtitle("Cross validated grid search results"))
# %%
# GBM
# create GBM estimator
xgb_mod = xgb.XGBRegressor()

kfold = KFold(n_splits=5, shuffle=True)

# define hyperparameters
hyper_grid = {
  'xgb_mod__n_estimators': [1000, 2500, 5000],
  'xgb_mod__learning_rate': [0.001, 0.01, 0.1],
  'xgb_mod__max_depth': [3, 5, 7, 9],
  'xgb_mod__min_child_weight': [1, 5, 15]
}

# create random search object
random_search = RandomizedSearchCV(
    xgb_mod, 
    param_distributions=hyper_grid, 
    n_iter=20, 
    cv=kfold, 
    scoring=lossFn, 
    n_jobs=-1, 
    random_state=13
)

# execute random search
random_search_results = random_search.fit(X_train, y_train)

# best model score
print(np.abs(random_search_results.best_score_))

# best hyperparameter values
print(random_search_results.best_params_)
# %%
