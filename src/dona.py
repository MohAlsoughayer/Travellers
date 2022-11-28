#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# %%
df = pd.read_csv('cleaned_data.csv')

# %%
df.shape

#%%
df.info()


#%%
df.head(3)


#%%
df.groupby('Airline').describe()["Price"]

#%%
df.groupby(['Airline','Total_Stops']).describe()["Price"]
# %%
# q1- who sold most tickets?
df['Airline'].value_counts()

#%%
# q1- (chart)WHICH AIRLINE HAS MOST FLIGHTS 
plt.subplots(figsize=(10,5))
sns.countplot(x="Airline", data=df, order = df['Airline'].value_counts().index)
plt.title("title")#, weight="bold",fontsize=20, pad=20)
plt.ylabel("count", weight="bold", fontsize=20)
plt.xlabel("Airline Name", weight="bold", fontsize=16)
plt.xticks(rotation= 45)
plt.show()





#%%
#so most flight have one stop and its through jet airways so that would explain why it has the highest value
#but did the majority of jet airways guests take the 1 stop flights or is it an outlier?


# q2 how many of jet airways guests took 1 stop flights 
df['Total_Stops'].value_counts()


#%%
fig = px.histogram(df, x='Airline',
             color='Total_Stops', barmode='group',
             
             title='title',
             labels={'',
                     '',
                     ''
                     })
fig.update_layout(yaxis_title='Number of flights')
fig.show()





#%%
# q3 why is one stop the most spent ? is it its price or is it that its that its cheap and a lot of poeple go for it?


question_3 = df.groupby(['Total_Stops', 'Airline']).describe()['Price']

# %%
question_3.drop(question_3.columns[[2,3,4,6]], axis=1, inplace=True)
  
# %%
question_3



#%%

#q4 do the diffrent types of additional info change the price?
df['Additional_Info'].unique()


# %%
plt.figure(figsize = (15, 10))
plt.title('Price VS Additional Information')
plt.scatter(df['Additional_Info'], df['Price'])
plt.xticks(rotation = 70)
plt.xlabel('additional info')#,weight="bold",fontsize=10, pad=20)
plt.ylabel('ticket price', pad = 30)#,weight="bold",fontsize=10, pad=2)




#%%
from sklearn.model_selection import train_test_split
 
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, roc_auc_score

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import math
# Minimize convergence warning messages
import warnings
warnings.filterwarnings("ignore")

# %%

df2 = pd.read_csv('repo/Travellers/data/final_data.csv')


#%%

# create train/test split
train, test = train_test_split(df2, train_size=0.7)

# get predictors and response varaibles for each set 
X_train = train.drop(["Price","year"], axis=1)
y_train = train[["Price"]]
X_test = test.drop(["Price","year"], axis=1)
y_test = test[["Price"]]





#%%


# Step 2: standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 3: create model object
en_mod = ElasticNet(alpha=1)

# Step 4: fit/train model
en_fit = en_mod.fit(X_train_scaled, y_train)


#%%
en_fit.coef_

#%%
pred = en_fit.predict(X_train)

#%%
# compute RMSE
lossfunc = 'neg_root_mean_squared_error'


#%%
kfold = KFold(n_splits=5, random_state=123, shuffle=True)

# Create grid of hyperparameter values
hyper_grid = {'alpha': 10**np.linspace(10, -5, 50)*0.5}

#%%

grid_search = GridSearchCV(en_mod, hyper_grid, cv=kfold, scoring=lossfunc)
results = grid_search.fit(X_train, y_train)

# Optimal penalty parameter in grid search
results.best_estimator_


#%%
# Best model's cross validated RMSE
round(abs(results.best_score_), 2)