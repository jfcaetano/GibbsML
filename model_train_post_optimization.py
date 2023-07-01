### ΔG ML modelling
### JFCAETANO 2023
### MIT Licence

import csv, math, sys
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn import ensemble
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


#Training - this outpus the performace of each model according with different algorithms

#select models
my_model = ["RandomForestRegressor(n_estimators= 250, min_samples_split= 2, min_samples_leaf= 1, max_features='sqrt', max_depth= None, bootstrap= False)", "GradientBoostingRegressor(n_estimators= 250, min_samples_split= 5, min_samples_leaf= 5, max_features= 'sqrt', max_depth= None)"]

#train:test:split
train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10

o = list()
#munber of trials
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for time in my_list:
    data = pd.read_csv('Model_Start.csv')

    exclude_cols=['Solvent Type','Solute Type','Solvent SMILES','Solute SMILES', 'dG', 'Solvent Key', 'Solute Key']
    ycols=['dG', 'Solvent Type']

    X_names=[x for x in data.columns if x not in exclude_cols]

    X = data.loc[:,X_names]
    X = X.fillna(0)
    y = data.loc[:,"dG"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    
    for mod in my_model:    
        model = eval(f"{mod}")
        model.fit(X_train, y_train)
        y_train_fitted=model.predict(X_train)
        y_test_fitted=model.predict(X_test)
        rsq_test = np.corrcoef(y_test,y_test_fitted)[0,1]**2
        Score_train = model.score(X_train, y_train)
        Score_test = model.score(X_test, y_test)
        MAE=mean_absolute_error(y_test,y_test_fitted)
        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=47)
        n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        STD=std(n_scores)
        MSE = np.square(np.subtract(y_test,y_test_fitted)).mean() 
        RMSE = math.sqrt(MSE)
    
        nl = dict()
        nl[f"Algorithm"]=f"{mod}"
        nl[f"Score_test"]=Score_test
        nl[f"Score_train"]=Score_train
        nl[f"MAE"]=MAE
        nl[f"RMSE"]=RMSE
        nl[f"STD"]=STD
        nl[f"R2"]=rsq_test

        o.append(nl)

output_fn = 'training_models_opt.csv'
with open(output_fn,'w',newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=o[0].keys())
    writer.writeheader()
    for new_row in o:
        writer.writerow(new_row)

