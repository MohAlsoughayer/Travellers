# Author: Mohammed Alsoughayer
# Description: Perform an ML regression model to predict the price of flights based on certain features
# %%
# Helper packages
import numpy as np
import pandas as pd
import math
from plotnine import ggplot, aes, geom_density, geom_line, geom_point, ggtitle, labs
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Modeling process
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# %%
# read data
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
# GBM, part 1 basic GBM to get optimal hyperparameters
# create GBM estimator
xgb_mod = xgb.XGBRegressor()

# create 5 fold CV object
kfold = KFold(n_splits=5, shuffle=True)

# create pre-processing pipeline
preprocessor = ColumnTransformer(
  remainder="passthrough",
  transformers=[
    ("scale", StandardScaler(), selector(dtype_include="number")),
    ("one-hot", OneHotEncoder(), selector(dtype_include="object"))
  ])

# create modeling pipeline
model_pipeline = Pipeline(steps=[
  ("preprocessor", preprocessor),
  ("xgb_mod", xgb_mod),
])

# define hyperparameters
hyper_grid = {
  'xgb_mod__n_estimators': [i for i in range(100,3001,100)],
  'xgb_mod__learning_rate': [0.001, 0.01, 0.1],
  'xgb_mod__max_depth': [i for i in range(1,41,2)],
  'xgb_mod__min_child_weight': [i for i in range(1,41,4)]
}

# create random search object
random_search = RandomizedSearchCV(
    model_pipeline, 
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
# GBM, part 2 stochastic GBM using optimal hyperparameters
# create GBM estimator with previous parameter settings
xgb_mod = xgb.XGBRegressor(
    n_estimators=random_search_results.best_score_['xgb_mod__n_estimators'],
    learning_rate=random_search_results.best_score_['xgb_mod__learning_rate'],
    max_depth=random_search_results.best_score_['xgb_mod__max_depth'],
    min_child_weight=random_search_results.best_score_['xgb_mod__min_child_weight']
)

# create modeling pipeline
model_pipeline = Pipeline(steps=[
  ("preprocessor", preprocessor),
  ("xgb_mod", xgb_mod),
])

# define stochastic hyperparameters
stochastic_hyper_grid = {
  'xgb_mod__subsample': [0.5, 0.75, 1],
  'xgb_mod__colsample_bytree': [0.5, 0.75, 1],
  'xgb_mod__colsample_bylevel': [0.5, 0.75, 1],
  'xgb_mod__colsample_bynode': [0.5, 0.75, 1]
}

stochastic_random_search = RandomizedSearchCV(
    model_pipeline, 
    param_distributions=stochastic_hyper_grid, 
    n_iter=20, 
    cv=kfold, 
    scoring=lossFn, 
    n_jobs=-1, 
    random_state=13
)

# execute random search
stochastic_random_search_results = stochastic_random_search.fit(X_train, y_train)

# best model score
print(np.abs(stochastic_random_search_results.best_score_))

# best hyperparameter values
print(stochastic_random_search_results.best_params_)
# %%
# feature interpetations of final model
# preprocess training data
X_encoded = preprocessor.fit_transform(X_train)

# create final model object
final_model = xgb.XGBRegressor(
    n_estimators=random_search_results.best_score_['xgb_mod__n_estimators'],
    learning_rate=random_search_results.best_score_['xgb_mod__learning_rate'],
    max_depth=random_search_results.best_score_['xgb_mod__max_depth'],
    min_child_weight=random_search_results.best_score_['xgb_mod__min_child_weight'],
    subsample=stochastic_random_search_results.best_params_['xgb_mod__subsample'],
    colsample_bytree=stochastic_random_search_results.best_params_['xgb_mod__colsample_bytree'],
    colsample_bylevel=stochastic_random_search_results.best_params_['xgb_mod__colsample_bylevel'],
    colsample_bynode=stochastic_random_search_results.best_params_['xgb_mod__colsample_bynode']
)

final_model_fit = final_model.fit(X_encoded, y_train)

# extract feature importances
vi = pd.DataFrame({'feature': preprocessor.get_feature_names_out(),
                   'importance': final_model_fit.feature_importances_})

# get top 20 influential features
top_20_features = vi.nlargest(20, 'importance')

# plot feature importance
(ggplot(top_20_features, aes(x='importance', y='reorder(feature, importance)'))
 + geom_point()
 + labs(y=None))


# %%
# Finally, Run test set features on the model and graph target predictions on top of actual values
# Transform Test set features 
X_test_encoded = preprocessor.transform(X_test)

# Run final model on transformed X
y_predicted = final_model_fit.predict(X_test_encoded)
ovservations = np.arange(1,3140)

# Plot target predicted and actual values
trace1 = go.Scatter(
    x=ovservations,
    y=y_predicted,
    name='target prediction',
    marker=dict(
        color='rgb(0,0,204)'
               )
)
trace2 = go.Scatter(
    x=ovservations,
    y=y_test.values.ravel(),
    name='target actual',
    marker=dict(
        color='rgb(204,0,0)'
               )
)
fig = go.Figure()
fig.add_traces([trace2,trace1])
fig.update_layout(title = 'Final Model Test Set Target Prediction & Actual', yaxis={'title':'Price of Ticket (Rupee)'}, xaxis={'title':'Flight Observation Number'})
fig.show()

# Print RMSE
print("Root mean squared error: ", math.sqrt(mean_squared_error(y_test.values.ravel(), y_predicted)))

# %%
