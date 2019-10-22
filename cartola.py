#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:38:56 2019

@author: nailson
"""

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


#Import data
df = pd.read_csv('cartola_2014.csv')

####################### Partial EDA ###############################

df.head()
# Total de na por colunas
df.isnull().sum()
df.describe()
df.clube_id.unique()

####################### END Partial EDA ###############################

####################### Data Preprocessing EDA ###############################

df = df.drop(columns=['atleta_id', 'partida_id'])

y_train = df.loc[:, 'nota'].values
df = df.drop(columns=['nota'])
    
preprocess = make_column_transformer(
        (StandardScaler(), ['rodada', 'participou', 'jogos_num', 'pontos_num', 'media_num',
          'preco_num', 'variacao_num', 'mando', 'titular', 'substituido', 'tempo_jogado',
          'FS', 'PE', 'A', 'FT', 'FD', 'FF', 'G', 'I','PP', 'RB', 'FC', 'GC', 'CA', 'CV', 'SG', 
          'DD', 'DP', 'GS']),
        (OneHotEncoder(categories='auto'), ['clube_id', 'posicao_id'])
    )
    
X_train = preprocess.fit_transform(df)

####################### END Data Preprocessing EDA ###############################


####################### Simple Linear Regression ###############################
classifier  =  LinearRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred))

####################### END Simple Linear Regression ###############################

####################### Regressao Ridge w/ cross validation ###############################
lambdas = np.logspace(-5, -1, 50)
ridge = RidgeCV(alphas = lambdas, fit_intercept=True, cv=10)
ridge.fit(X_train, y_train)

ridge.alpha_
ridge.cv_values_
ridge.score(X_train, y_train)
ridge.get_params()
ridge.coef_

print('Melhor lamda: %0.5f' %ridge.alpha_)

important_variables = []
club_plus_position = 0
for i in range(len(ridge.coef_)):
    if abs(ridge.coef_[i]) >= 0.1:
        if(i < len(df.drop(columns=['clube_id', 'posicao_id']).columns)):
            important_variables.append(df.drop(columns=['clube_id', 'posicao_id']).columns[i])
        else:
            club_plus_position += 1
print('Variáveis mais importantes para o Ridge: ', important_variables)
print('Juntamente com %d variáveis referentes a posição do jogador e id do clube' %(club_plus_position))


y_pred = ridge.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred))
####################### END Regressao Ridge w/ cross validation ###############################


####################### Regressao LASSO w/ cross validation ###############################
lasso = LassoCV(cv=10, alphas=lambdas)
sfm = SelectFromModel(lasso, prefit=True)
sfm.fit(X_train, y_train)
n_features = sfm.transform(X_train)
indices = sfm.get_support(indices=True)

lasso.fit(X_train, y_train)
lasso.coef_

y_pred = lasso.predict(X_train[:, indices])

math.sqrt(mean_squared_error(y_train, y_pred))

####################### END Regressao LASSO w/ cross validation ###############################


####################### KNN w/ cross validation ###############################
knn = KNeighborsRegressor()
param_grid = {'n_neighbors': np.arange(1, 16)}
gscv = GridSearchCV(knn, param_grid, cv=10, verbose=10, n_jobs=-1)
gscv.fit(X_train, y_train)
gscv.best_params_
gscv.best_score_
gscv.best_estimator_

y_pred = gscv.predict(X_train)

math.sqrt(mean_squared_error(y_train, y_pred))

####################### END KNN w/ cross validation ###############################